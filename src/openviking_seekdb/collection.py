"""OpenViking ICollection implementation backed by OceanBase SeekDB."""

from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from openviking.storage.vectordb.collection.collection import ICollection
from openviking.storage.vectordb.collection.result import (
    AggregateResult,
    DataItem,
    FetchDataInCollectionResult,
    SearchItemResult,
    SearchResult,
    UpsertDataResult,
)
from openviking.storage.vectordb.index.index import IIndex

from .client import SeekDBClient
from .config import SeekDBConfig
from .errors import SeekDBCollectionMissingError, SeekDBUnsupportedError
from .filters import SeekDBFilterCompiler, SQLFilter
from .schema import (
    SchemaSpec,
    derive_index_name,
    normalize_vector,
    validate_identifier,
)
from .sparse import SparseVectorCodec

logger = logging.getLogger(__name__)

_SEEKDB_VECTOR_LIMIT = 16_384
_DEFAULT_INDEX_NAME = "default"


@dataclass(frozen=True)
class _DecodedRow:
    id: str
    metadata: dict[str, Any]
    content: Optional[str]
    embedding: Optional[list[float]]
    sparse_embedding: Optional[str]
    sparse_terms: Optional[dict[str, float]]

    def to_record(self, schema: SchemaSpec, output_fields: Optional[list[str]]) -> dict[str, Any]:
        include_all = not output_fields
        requested = set(output_fields or []) | {schema.primary_key}
        result: dict[str, Any] = {schema.primary_key: self.id}
        for name, spec in schema.fields.items():
            if name == schema.primary_key:
                continue
            if not include_all and name not in requested:
                continue
            if name == schema.vector_field:
                result[name] = list(self.embedding) if self.embedding is not None else None
            elif name == schema.sparse_vector_field:
                result[name] = (
                    dict(self.sparse_terms or {}) if self.sparse_terms is not None else {}
                )
            elif name == schema.content_field:
                result[name] = self.content if self.content is not None else ""
            elif name in self.metadata:
                result[name] = self.metadata[name]
            elif spec.default_value is not None:
                result[name] = spec.default_value
        return result


@dataclass(frozen=True)
class _RankedRow:
    row: _DecodedRow
    score: float
    origin_score: Optional[float] = None
    time_score: Optional[float] = None


@dataclass(frozen=True)
class _TimeDecaySpec:
    field: str
    origin_ms: float
    offset_ms: float
    scale_ms: float
    decay: float


class SeekDBIndex(IIndex):
    """Index facade returned to OpenViking; storage stays in the collection table."""

    def __init__(self, collection: "SeekDBCollection", index_name: str, meta: Mapping[str, Any]):
        super().__init__(meta=dict(meta))
        self.collection = collection
        self.index_name = index_name
        self.meta = dict(meta)

    def upsert_data(self, delta_list: List[Any]):
        records = []
        for delta in delta_list:
            if isinstance(delta, Mapping):
                records.append(dict(delta))
                continue
            fields = getattr(delta, "fields", None)
            if isinstance(fields, str):
                fields = json.loads(fields)
            if not isinstance(fields, Mapping):
                raise ValueError("SeekDB index delta must contain JSON fields")
            record = dict(fields)
            if getattr(delta, "label", None) is not None:
                record.setdefault("id", delta.label)
            vector = getattr(delta, "vector", None)
            if vector is not None:
                record.setdefault(self.collection.schema.vector_field, vector)
            records.append(record)
        self.collection.upsert_data(records)

    def delete_data(self, delta_list: List[Any]):
        ids = []
        for delta in delta_list:
            label = delta.get("id") if isinstance(delta, Mapping) else getattr(delta, "label", None)
            if label is not None:
                ids.append(label)
        self.collection.delete_data(ids)

    def search(
        self,
        query_vector: Optional[List[float]],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        sparse_raw_terms: Optional[List[str]] = None,
        sparse_values: Optional[List[float]] = None,
    ) -> Tuple[List[Any], List[float]]:
        if sparse_raw_terms or sparse_values:
            if len(sparse_raw_terms or []) != len(sparse_values or []):
                raise ValueError("sparse_raw_terms and sparse_values must have equal length")
            sparse_vector = dict(zip(sparse_raw_terms or [], sparse_values or [], strict=True))
        else:
            sparse_vector = None
        result = self.collection.search_by_vector(
            self.index_name,
            dense_vector=query_vector,
            sparse_vector=sparse_vector,
            limit=limit,
            filters=filters,
            output_fields=["id"],
        )
        return [item.id for item in result.data], [float(item.score or 0.0) for item in result.data]

    def aggregate(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.collection.aggregate_data(self.index_name, filters=filters).agg

    def update(
        self,
        scalar_index: Optional[List[str] | Dict[str, Any]],
        description: Optional[str],
    ):
        names = list(scalar_index) if isinstance(scalar_index, list) else scalar_index
        updated = self.collection.update_index(
            self.index_name,
            scalar_index=names if isinstance(names, list) else None,
            description=description,
        )
        self.meta = updated

    def rebuild_scalar_index(
        self, scalar_index: List[str], cands_fields: Iterable[Tuple[int, str]]
    ) -> None:
        # SeekDB filters over the JSON metadata column. The logical scalar-field
        # declaration remains valuable to OpenViking, while a data copy is not.
        self.update(scalar_index, None)

    def get_meta_data(self):
        return self.collection.get_index_meta_data(self.index_name) or {}

    def close(self):
        pass

    def drop(self):
        self.collection.drop_index(self.index_name)

    def get_newest_version(self):
        return int(time.time() * 1000)

    def need_rebuild(self) -> bool:
        return False


class SeekDBCollection(ICollection):
    """A single OpenViking collection/table in an OceanBase SeekDB server."""

    def __init__(
        self,
        *,
        client: SeekDBClient,
        config: SeekDBConfig,
        logical_name: str,
        physical_name: str,
        schema: SchemaSpec,
        sparse_weight: float,
        index_meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        validate_identifier(physical_name, kind="physical collection table name")
        self.client = client
        self.config = config
        self.logical_name = logical_name
        self.physical_name = physical_name
        self.schema = schema
        self.sparse_weight = float(sparse_weight)
        if not 0 <= self.sparse_weight <= 1:
            raise ValueError("sparse_weight must be in [0, 1]")
        self._indexes: dict[str, dict[str, Any]] = {}
        if index_meta and index_meta.get("IndexName"):
            self._indexes[str(index_meta["IndexName"])] = dict(index_meta)
        self._filter_compiler = SeekDBFilterCompiler(schema)
        self._sparse_codec = SparseVectorCodec(
            max_dimension=config.sparse_max_dimension,
            max_nonzero=config.sparse_max_nonzero,
        )
        self._mutation_lock = threading.RLock()
        self._bulk_ingest_depth = 0
        self._refresh_pending = False
        self._closed = False

    # ------------------------------------------------------------------
    # DDL / catalog lifecycle
    # ------------------------------------------------------------------

    def create_physical_table(self) -> None:
        table = _quote(self.physical_name)
        with self.client.transaction() as connection, connection.cursor() as cursor:
            if self.client.table_exists(self.physical_name, cursor=cursor):
                raise RuntimeError(f"SeekDB physical table {self.physical_name!r} already exists")
            cursor.execute(
                f"CREATE TABLE {table} ("
                "`id` VARCHAR(512) NOT NULL, "
                "`metadata` JSON NOT NULL, "
                "`content` LONGTEXT, "
                f"`embedding` VECTOR({self.schema.dimension}), "
                "`sparse_embedding` SPARSEVECTOR, "
                "`sparse_terms` JSON, "
                "PRIMARY KEY (`id`)"
                ") ORGANIZATION = HEAP DEFAULT CHARSET = utf8mb4"
            )

    def drop_physical_table(self) -> None:
        with self.client.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {_quote(self.physical_name)}")

    def _physical_index_names(self, index_name: str) -> dict[str, str]:
        return {
            "vector": derive_index_name(self.logical_name, index_name, "vector"),
            "sparse": derive_index_name(self.logical_name, index_name, "sparse"),
            "fulltext": derive_index_name(self.logical_name, index_name, "fulltext"),
        }

    def _index_is_physically_ready(self, meta: Mapping[str, Any]) -> bool:
        current = self.client.table_indexes(self.physical_name)
        internal = meta.get("_seekdb", {})
        vector_index_type = str(internal.get("vector_index_type", "hnsw"))
        vector_name = internal.get("vector_index")
        if vector_index_type != "flat" and (
            not isinstance(vector_name, str) or vector_name not in current
        ):
            return False
        if internal.get("sparse_enabled") and internal.get("sparse_index") not in current:
            return False
        return not (
            internal.get("fulltext_enabled") and internal.get("fulltext_index") not in current
        )

    def _create_vector_index(self, index_name: str, meta: Mapping[str, Any]) -> Optional[str]:
        index_type = self.config.vector_index_type
        if index_type == "flat":
            return None
        physical = self._physical_index_names(index_name)["vector"]
        current = self.client.table_indexes(self.physical_name)
        if physical in current:
            return physical
        vector_meta = meta.get("VectorIndex", {})
        distance = str(vector_meta.get("Distance", "cosine")).lower()
        if distance == "ip":
            distance = "inner_product"
        if distance not in {"cosine", "l2", "inner_product"}:
            raise ValueError(f"Unsupported SeekDB distance metric {distance!r}")
        if index_type == "hnsw_bq" and distance == "inner_product":
            raise ValueError("SeekDB hnsw_bq supports cosine or l2 distance only")

        if index_type.startswith("hnsw"):
            params = {
                "DISTANCE": distance,
                "TYPE": index_type,
                "LIB": "vsag",
                "M": self.config.hnsw_m,
                "EF_CONSTRUCTION": self.config.hnsw_ef_construction,
                "EF_SEARCH": self.config.hnsw_ef_search,
            }
            if self.config.hnsw_extra_info_max_size is not None:
                params["EXTRA_INFO_MAX_SIZE"] = self.config.hnsw_extra_info_max_size
            if index_type == "hnsw_bq":
                if self.config.hnsw_refine_k is not None:
                    params["REFINE_K"] = self.config.hnsw_refine_k
                if self.config.hnsw_refine_type is not None:
                    params["REFINE_TYPE"] = self.config.hnsw_refine_type.upper()
                if self.config.hnsw_bq_bits_query is not None:
                    params["BQ_BITS_QUERY"] = self.config.hnsw_bq_bits_query
                if self.config.hnsw_bq_use_fht is not None:
                    params["BQ_USE_FHT"] = str(self.config.hnsw_bq_use_fht).lower()
        else:
            params = {
                "DISTANCE": distance,
                "TYPE": index_type,
                "LIB": "ob",
                "NLIST": self.config.ivf_nlist,
                "SAMPLE_PER_NLIST": self.config.ivf_sample_per_nlist,
            }
            if index_type == "ivf_pq":
                pq_m = self.config.ivf_pq_m or _default_pq_subquantizers(self.schema.dimension)
                if self.schema.dimension % pq_m:
                    raise ValueError(
                        "ivf_pq_m must divide the vector dimension; "
                        f"dimension={self.schema.dimension}, ivf_pq_m={pq_m}"
                    )
                params["NBITS"] = self.config.ivf_nbits
                params["M"] = pq_m
        option_sql = ", ".join(f"{key}={value}" for key, value in params.items())
        with self.client.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"CREATE VECTOR INDEX {_quote(physical)} ON {_quote(self.physical_name)} "
                f"(`embedding`) WITH ({option_sql})"
            )
        return physical

    def _create_sparse_index(self, index_name: str) -> str:
        if self.schema.sparse_vector_field is None:
            raise ValueError("sparse_weight > 0 requires a sparse_vector schema field")
        physical = self._physical_index_names(index_name)["sparse"]
        current = self.client.table_indexes(self.physical_name)
        if physical in current:
            return physical
        options = ["DISTANCE=inner_product", "TYPE=sindi", "LIB=vsag"]
        if self.config.sparse_prune:
            options.extend(
                [
                    "PRUNE=true",
                    f"REFINE={'true' if self.config.sparse_refine else 'false'}",
                ]
            )
        with self.client.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"CREATE VECTOR INDEX {_quote(physical)} ON {_quote(self.physical_name)} "
                f"(`sparse_embedding`) WITH ({', '.join(options)})"
            )
        return physical

    def _create_fulltext_index(self, index_name: str) -> str:
        if not self.config.fulltext:
            raise SeekDBUnsupportedError("full-text indexing is disabled by configuration")
        if self.schema.content_field is None:
            raise ValueError("SeekDB full-text index requires a content text field")
        physical = self._physical_index_names(index_name)["fulltext"]
        current = self.client.table_indexes(self.physical_name)
        if physical in current:
            return physical
        analyzer = self.config.fulltext_analyzer.upper()
        clause = f" WITH PARSER {analyzer}"
        if self.config.fulltext_properties:
            values = ", ".join(
                f"{validate_identifier(str(key), kind='fulltext property')}="
                f"{_sql_string(str(value))}"
                for key, value in self.config.fulltext_properties.items()
            )
            clause += f" PARSER_PROPERTIES=({values})"
        with self.client.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"CREATE FULLTEXT INDEX {_quote(physical)} ON {_quote(self.physical_name)} "
                f"(`{self.schema.content_field}`){clause}"
            )
        return physical

    def create_index(self, index_name: str, meta_data: Dict[str, Any]) -> IIndex:
        validate_identifier(index_name, kind="logical index name")
        if index_name in self._indexes:
            raise ValueError(f"SeekDB index {index_name!r} already exists")
        meta = dict(meta_data)
        meta.setdefault("IndexName", index_name)
        existing_before = set(self.client.table_indexes(self.physical_name))
        names: dict[str, str] = {}
        created_names: list[str] = []
        try:
            vector_name = self._create_vector_index(index_name, meta)
            if vector_name is not None:
                names["vector"] = vector_name
                if vector_name not in existing_before:
                    created_names.append(vector_name)
            sparse_enabled = self.sparse_weight > 0.0
            if sparse_enabled:
                names["sparse"] = self._create_sparse_index(index_name)
                if names["sparse"] not in existing_before:
                    created_names.append(names["sparse"])
            fulltext_enabled = bool(
                self.config.fulltext
                and self.schema.content_field
                and self.schema.raw.get("FullText", None) is not False
            )
            if fulltext_enabled:
                names["fulltext"] = self._create_fulltext_index(index_name)
                if names["fulltext"] not in existing_before:
                    created_names.append(names["fulltext"])
        except Exception:
            for physical in created_names:
                try:
                    self._drop_physical_index(physical)
                except Exception:
                    logger.exception(
                        "Failed to clean up partially created SeekDB index %s", physical
                    )
            raise

        vector_meta = dict(meta.get("VectorIndex") or {})
        vector_meta["IndexType"] = self.config.vector_index_type
        vector_meta["Dimension"] = self.schema.dimension
        vector_meta["EnableSparse"] = self.sparse_weight > 0.0
        if self.sparse_weight > 0.0:
            vector_meta["SearchWithSparseLogitAlpha"] = self.sparse_weight
        scalar_index = list(dict.fromkeys(str(item) for item in meta.get("ScalarIndex", [])))
        stored = {
            **meta,
            "IndexName": index_name,
            "VectorIndex": vector_meta,
            "ScalarIndex": scalar_index,
            "Description": str(meta.get("Description", "")),
            "_seekdb": {
                "vector_index_type": self.config.vector_index_type,
                "vector_index": names.get("vector"),
                "sparse_enabled": self.sparse_weight > 0.0,
                "sparse_index": names.get("sparse"),
                "fulltext_enabled": "fulltext" in names,
                "fulltext_index": names.get("fulltext"),
                "fulltext_analyzer": self.config.fulltext_analyzer,
            },
        }
        self.client.save_index(self.logical_name, index_name, stored)
        self._indexes[index_name] = stored
        return SeekDBIndex(self, index_name, stored)

    def has_index(self, index_name: str) -> bool:
        meta = self._indexes.get(index_name)
        return bool(meta and self._index_is_physically_ready(meta))

    def get_index(self, index_name: str) -> Optional[IIndex]:
        meta = self._indexes.get(index_name)
        if not meta or not self._index_is_physically_ready(meta):
            return None
        return SeekDBIndex(self, index_name, meta)

    def list_indexes(self):
        return list(self._indexes)

    def get_index_meta_data(self, index_name: str):
        meta = self._indexes.get(index_name)
        if not meta:
            return None
        public = _public_index_meta(meta, self.schema)
        public["Statistics"] = {"DocCount": self.aggregate_data(index_name).agg.get("_total", 0)}
        return public

    def _drop_physical_index(self, index_name: str) -> None:
        with self.client.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(f"DROP INDEX {_quote(index_name)} ON {_quote(self.physical_name)}")

    def drop_index(self, index_name: str):
        meta = self._indexes.get(index_name)
        if not meta:
            return
        internal = meta.get("_seekdb", {})
        for key in ("vector_index", "sparse_index", "fulltext_index"):
            physical = internal.get(key)
            if physical in self.client.table_indexes(self.physical_name):
                self._drop_physical_index(str(physical))
        self.client.delete_index(self.logical_name, index_name)
        self._indexes.pop(index_name, None)

    def update_index(
        self,
        index_name: str,
        scalar_index: Optional[List[str]] = None,
        description: Optional[str] = None,
    ):
        meta = self._indexes.get(index_name)
        if not meta:
            return self.get_index_meta_data(index_name)
        updated = dict(meta)
        if scalar_index is not None:
            unknown = [field for field in scalar_index if field not in self.schema.fields]
            if unknown:
                raise ValueError(f"unknown ScalarIndex fields: {unknown}")
            updated["ScalarIndex"] = list(dict.fromkeys(scalar_index))
        if description is not None:
            updated["Description"] = description
        self.client.save_index(self.logical_name, index_name, updated)
        self._indexes[index_name] = updated
        return _public_index_meta(updated, self.schema)

    def update(self, fields: Optional[Dict[str, Any]] = None, description: Optional[str] = None):
        with self._mutation_lock:
            if isinstance(fields, list):
                updates = fields
            elif isinstance(fields, dict):
                updates = fields.get("Fields", [fields] if fields else [])
            elif fields is None:
                updates = None
            else:
                raise ValueError("SeekDB collection fields must be a list or object")
            updated = self.schema.with_updates(new_fields=updates, description=description)
            self.client.save_collection(self.logical_name, self.physical_name, updated.to_dict())
            self.schema = updated
            self._filter_compiler = SeekDBFilterCompiler(updated)
            return True

    def get_meta_data(self, *, raise_on_error: bool = False):
        del raise_on_error
        raw = self.schema.to_dict()
        raw["CollectionName"] = self.logical_name
        raw["ScalarIndex"] = list(self.schema.scalar_index)
        physical = self.client.table_indexes(self.physical_name)
        fulltext_specs = []
        for meta in self._indexes.values():
            internal = meta.get("_seekdb", {})
            if internal.get("fulltext_enabled") and internal.get("fulltext_index") in physical:
                fulltext_specs.append(
                    {
                        "Field": self.schema.content_field,
                        "Analyzer": {"Tokenizer": internal.get("fulltext_analyzer", "ik")},
                        "Name": internal.get("fulltext_index"),
                    }
                )
        if fulltext_specs:
            raw["FullText"] = fulltext_specs
        else:
            raw.pop("FullText", None)
        return raw

    def close(self):
        # The adapter owns and closes the shared connection pool. Collection
        # wrappers may be garbage-collected at arbitrary times.
        pass

    def drop(self):
        for index_name in list(self._indexes):
            self.drop_index(index_name)
        self.drop_physical_table()
        self.client.delete_collection_record(self.logical_name)
        self._indexes.clear()

    # ------------------------------------------------------------------
    # Record conversion and writes
    # ------------------------------------------------------------------

    def _encoded_vector(self, value: Optional[Sequence[float]]) -> Optional[str]:
        if value is None:
            return None
        normalized = normalize_vector(value, self.schema.dimension)
        return "[" + ",".join(format(component, ".17g") for component in normalized) + "]"

    def _encode_sparse(self, value: Any) -> tuple[Optional[str], Optional[str]]:
        if value is None:
            return None, None
        normalized = self._sparse_codec.normalize(value)
        if not normalized:
            return None, json.dumps({}, ensure_ascii=False)
        sparse_sql = self._sparse_codec.encode(normalized)
        raw = {str(key): float(weight) for key, weight in dict(value).items()}
        return sparse_sql, json.dumps(raw, ensure_ascii=False, allow_nan=False)

    def _prepare_full_record(self, record: Mapping[str, Any]) -> tuple[Any, ...]:
        primary = str(record.get(self.schema.primary_key, "")).strip()
        if not primary or len(primary) > 512 or "\x00" in primary:
            raise ValueError("SeekDB record requires a non-empty id of at most 512 characters")
        metadata = self.schema.encode_metadata(record)
        content = self.schema.content_value(record)
        vector = self._encoded_vector(self.schema.vector_value(record))
        sparse_value = (
            record.get(self.schema.sparse_vector_field) if self.schema.sparse_vector_field else None
        )
        sparse_sql, sparse_terms = self._encode_sparse(sparse_value)
        return (
            primary,
            json.dumps(metadata, ensure_ascii=False, allow_nan=False),
            content,
            vector,
            sparse_sql,
            sparse_terms,
        )

    def _write_full_rows(self, cursor, rows: Sequence[tuple[Any, ...]]) -> None:
        sql = (
            f"INSERT INTO {_quote(self.physical_name)} "
            "(`id`, `metadata`, `content`, `embedding`, `sparse_embedding`, `sparse_terms`) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE "
            "`metadata` = %s, `content` = %s, `embedding` = %s, "
            "`sparse_embedding` = %s, `sparse_terms` = %s"
        )
        for row in rows:
            cursor.execute(sql, (*row, *row[1:]))

    def upsert_data(self, data_list: List[Dict[str, Any]], ttl=0):
        if ttl not in (0, None):
            raise SeekDBUnsupportedError("SeekDB adapter TTL writes are not implemented")
        if not data_list:
            return UpsertDataResult()
        rows = [self._prepare_full_record(record) for record in data_list]
        with self._mutation_lock:
            with self.client.transaction() as connection, connection.cursor() as cursor:
                self._write_full_rows(cursor, rows)
            self._request_index_refresh()
        return UpsertDataResult(ids=[row[0] for row in rows])

    def update_data(self, data_list: List[Dict[str, Any]]):
        if not data_list:
            return UpsertDataResult()
        prepared = []
        for record in data_list:
            primary = record.get(self.schema.primary_key)
            if primary is None:
                raise ValueError("SeekDB update requires id")
            primary = str(primary)
            unknown = sorted(set(record) - set(self.schema.fields) - {"_id"})
            if unknown:
                raise ValueError(f"SeekDB update contains unknown fields: {unknown}")
            patch = dict(record)
            patch.pop("_id", None)
            prepared.append((primary, patch))

        updated_ids: list[str] = []
        with self._mutation_lock:
            with self.client.transaction() as connection, connection.cursor() as cursor:
                for primary, patch in prepared:
                    cursor.execute(
                        f"SELECT `metadata`, `content`, `embedding`, `sparse_embedding`, "
                        f"`sparse_terms` FROM {_quote(self.physical_name)} "
                        "WHERE `id` = %s FOR UPDATE",
                        (primary,),
                    )
                    existing = cursor.fetchone()
                    if not existing:
                        raise ValueError(f"record not found for primary key(s): [{primary!r}]")
                    merged = _loads(existing["metadata"])
                    if self.schema.content_field and existing.get("content") is not None:
                        merged[self.schema.content_field] = existing["content"]
                    if existing.get("embedding") is not None:
                        merged[self.schema.vector_field] = _decode_vector(existing["embedding"])
                    if self.schema.sparse_vector_field and existing.get("sparse_terms") is not None:
                        merged[self.schema.sparse_vector_field] = _loads(existing["sparse_terms"])
                    merged[self.schema.primary_key] = primary
                    merged.update(patch)
                    row = self._prepare_full_record(merged)
                    cursor.execute(
                        f"UPDATE {_quote(self.physical_name)} SET `metadata` = %s, "
                        "`content` = %s, `embedding` = %s, `sparse_embedding` = %s, "
                        "`sparse_terms` = %s WHERE `id` = %s",
                        (*row[1:], primary),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError(
                            f"SeekDB update did not affect exactly one row: {primary}"
                        )
                    updated_ids.append(primary)
            self._request_index_refresh()
        return UpsertDataResult(ids=updated_ids)

    def fetch_data(self, primary_keys: List[Any]):
        if not primary_keys:
            return FetchDataInCollectionResult()
        ids = [str(item) for item in primary_keys]
        placeholders = ", ".join(["%s"] * len(ids))
        with self.client.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT `id`, `metadata`, `content`, `embedding`, "
                f"`sparse_embedding`, `sparse_terms` FROM {_quote(self.physical_name)} "
                f"WHERE `id` IN ({placeholders})",
                tuple(ids),
            )
            rows = [_decode_row(row) for row in cursor.fetchall()]
        by_id = {row.id: row for row in rows}
        items = []
        missing = []
        for primary in ids:
            row = by_id.get(primary)
            if row is None:
                missing.append(primary)
            else:
                items.append(DataItem(id=row.id, fields=row.to_record(self.schema, None)))
        return FetchDataInCollectionResult(items=items, ids_not_exist=missing)

    def delete_data(self, primary_keys: List[Any]):
        if not primary_keys:
            return
        ids = [str(item) for item in primary_keys]
        placeholders = ", ".join(["%s"] * len(ids))
        with self._mutation_lock:
            with self.client.transaction() as connection, connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {_quote(self.physical_name)} WHERE `id` IN ({placeholders})",
                    tuple(ids),
                )
            self._request_index_refresh()

    def delete_all_data(self):
        with self._mutation_lock:
            with self.client.transaction() as connection, connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {_quote(self.physical_name)}")
            self._request_index_refresh()

    def begin_bulk_ingest(self) -> None:
        with self._mutation_lock:
            self._bulk_ingest_depth += 1

    def end_bulk_ingest(self) -> None:
        with self._mutation_lock:
            if self._bulk_ingest_depth <= 0:
                raise RuntimeError("SeekDB end_bulk_ingest called without begin_bulk_ingest")
            self._bulk_ingest_depth -= 1
            if self._bulk_ingest_depth == 0:
                self._request_index_refresh()

    def _request_index_refresh(self) -> None:
        if self._bulk_ingest_depth > 0:
            self._refresh_pending = True
            return
        self.client.refresh_indexes()
        self._refresh_pending = False

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_search(limit: int, offset: int, output_fields: Optional[list[str]]) -> None:
        if type(limit) is not int or type(offset) is not int or limit < 0 or offset < 0:
            raise ValueError("SeekDB limit and offset must be non-negative integers")
        if output_fields is not None and (
            not isinstance(output_fields, list)
            or any(not isinstance(field, str) for field in output_fields)
        ):
            raise ValueError("output_fields must be a list of strings or null")

    def _require_index(self, index_name: str) -> dict[str, Any]:
        meta = self._indexes.get(index_name)
        if not meta:
            raise SeekDBCollectionMissingError(f"SeekDB index {index_name!r} does not exist")
        if not self._index_is_physically_ready(meta):
            raise SeekDBCollectionMissingError(
                f"SeekDB index {index_name!r} exists in the catalog but its physical index is missing"
            )
        return meta

    def _validate_output_fields(self, output_fields: Optional[list[str]]) -> None:
        if not output_fields:
            return
        unknown = sorted(set(output_fields) - set(self.schema.fields))
        if unknown:
            raise ValueError(f"unknown SeekDB output fields: {unknown}")

    def _select_dense_candidates(
        self,
        *,
        index_name: str,
        query_vector: Sequence[float],
        filters: SQLFilter,
        limit: int,
    ) -> list[_RankedRow]:
        budget = min(max(limit, 1), _SEEKDB_VECTOR_LIMIT)
        normalized = normalize_vector(
            query_vector, self.schema.dimension, field_name="query vector"
        )
        vector_sql = "[" + ",".join(format(component, ".17g") for component in normalized) + "]"
        meta = self._require_index(index_name)
        distance = str(meta.get("VectorIndex", {}).get("Distance", "cosine")).lower()
        if distance == "ip":
            distance = "inner_product"
        if distance == "cosine":
            distance_func = "cosine_distance"
            order = "ASC"
        elif distance == "l2":
            distance_func = "l2_distance"
            order = "ASC"
        elif distance == "inner_product":
            distance_func = "inner_product"
            order = "DESC"
        else:
            raise ValueError(f"Unsupported SeekDB distance metric {distance!r}")
        distance_expr = f"{distance_func}(r.`embedding`, %s)"
        where = f"WHERE {filters.sql}" if filters.sql != "TRUE" else ""
        inner_params = (vector_sql, *filters.params, vector_sql, budget)
        internal = meta.get("_seekdb", {})
        index_type = str(internal.get("vector_index_type", "hnsw"))
        approximate_clause = "" if index_type == "flat" else " APPROXIMATE"
        sql = (
            "SELECT id, metadata, content, embedding, sparse_embedding, sparse_terms, "
            "_ov_distance FROM ("
            "SELECT r.`id` AS id, r.`metadata` AS metadata, r.`content` AS content, "
            f"r.`embedding` AS embedding, r.`sparse_embedding` AS sparse_embedding, "
            f"r.`sparse_terms` AS sparse_terms, {distance_expr} AS _ov_distance "
            f"FROM {_quote(self.physical_name)} r {where} "
            f"ORDER BY {distance_expr} {order}{approximate_clause} LIMIT %s"
            ") AS hits ORDER BY _ov_distance "
            f"{order}, id ASC"
        )
        with self.client.transaction() as connection, connection.cursor() as cursor:
            if index_type.startswith("ivf") and self.config.ivf_nprobes is not None:
                cursor.execute(f"SET SESSION ob_ivf_nprobes = {int(self.config.ivf_nprobes)}")
            cursor.execute(sql, inner_params)
            rows = cursor.fetchall()
        return [
            _RankedRow(_decode_row(row), _distance_to_similarity(distance, row["_ov_distance"]))
            for row in rows
        ]

    def _select_sparse_candidates(
        self,
        *,
        index_name: str,
        sparse_vector: Mapping[str, Any],
        filters: SQLFilter,
        limit: int,
    ) -> list[_RankedRow]:
        meta = self._require_index(index_name)
        internal = meta.get("_seekdb", {})
        if not internal.get("sparse_enabled"):
            raise SeekDBUnsupportedError(
                "sparse query was supplied, but this SeekDB index does not enable sparse retrieval"
            )
        sparse_sql = self._sparse_codec.encode(sparse_vector)
        if not sparse_sql:
            return []
        budget = min(max(limit, 1), _SEEKDB_VECTOR_LIMIT)
        distance_expr = "negative_inner_product(r.`sparse_embedding`, %s)"
        where = f"WHERE {filters.sql}" if filters.sql != "TRUE" else ""
        sql = (
            "SELECT id, metadata, content, embedding, sparse_embedding, sparse_terms, "
            "_ov_distance FROM ("
            "SELECT r.`id` AS id, r.`metadata` AS metadata, r.`content` AS content, "
            f"r.`embedding` AS embedding, r.`sparse_embedding` AS sparse_embedding, "
            f"r.`sparse_terms` AS sparse_terms, {distance_expr} AS _ov_distance "
            f"FROM {_quote(self.physical_name)} r {where} "
            f"ORDER BY {distance_expr} ASC APPROXIMATE LIMIT %s"
            ") AS hits ORDER BY _ov_distance ASC, id ASC"
        )
        params = (sparse_sql, *filters.params, sparse_sql, budget)
        with self.client.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [_RankedRow(_decode_row(row), -float(row["_ov_distance"] or 0.0)) for row in rows]

    def search_by_vector(
        self,
        index_name: str,
        dense_vector: Optional[List[float]] = None,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        sparse_vector: Optional[Dict[str, float]] = None,
        output_fields: Optional[List[str]] = None,
        advance: Optional[Dict[str, Any]] = None,
        return_detail_info: bool = False,
    ) -> SearchResult:
        self._validate_search(limit, offset, output_fields)
        self._validate_output_fields(output_fields)
        if limit == 0:
            return SearchResult()
        if not dense_vector and not sparse_vector:
            return SearchResult()
        self._require_index(index_name)
        compiled = self._filter_compiler.compile(filters)
        decay = _parse_time_decay(advance, self.schema)
        requested = limit + offset
        hybrid = bool(dense_vector and sparse_vector)
        budget = (
            min(max(self.config.candidate_limit, requested), _SEEKDB_VECTOR_LIMIT)
            if hybrid or decay is not None
            else requested
        )
        if budget > _SEEKDB_VECTOR_LIMIT:
            raise ValueError(f"SeekDB vector limit+offset cannot exceed {_SEEKDB_VECTOR_LIMIT}")
        dense_rows = (
            self._select_dense_candidates(
                index_name=index_name,
                query_vector=dense_vector,
                filters=compiled,
                limit=budget,
            )
            if dense_vector
            else []
        )
        sparse_rows = (
            self._select_sparse_candidates(
                index_name=index_name,
                sparse_vector=sparse_vector,
                filters=compiled,
                limit=budget,
            )
            if sparse_vector
            else []
        )
        ranked = self._fuse_candidates(dense_rows, sparse_rows, decay)
        page = ranked[offset : offset + limit]
        result = SearchResult()
        for ranked_row in page:
            record = ranked_row.row.to_record(self.schema, output_fields)
            result.data.append(
                SearchItemResult(
                    id=ranked_row.row.id,
                    fields=record,
                    score=ranked_row.score,
                    origin_score=ranked_row.origin_score if return_detail_info else None,
                    addition_score=ranked_row.time_score if return_detail_info else None,
                )
            )
        return result

    def _fuse_candidates(
        self,
        dense_rows: Sequence[_RankedRow],
        sparse_rows: Sequence[_RankedRow],
        decay: Optional[_TimeDecaySpec],
    ) -> list[_RankedRow]:
        alpha = self.sparse_weight
        dense = {row.row.id: row for row in dense_rows}
        sparse = {row.row.id: row for row in sparse_rows}
        ids = set(dense) | set(sparse)
        fused: list[_RankedRow] = []
        for primary in ids:
            dense_row = dense.get(primary)
            sparse_row = sparse.get(primary)
            if dense_row and sparse_row:
                base_score = (1.0 - alpha) * dense_row.score + alpha * sparse_row.score
            elif dense_row:
                base_score = dense_row.score if not sparse_rows else (1.0 - alpha) * dense_row.score
            elif sparse_row:
                base_score = sparse_row.score if not dense_rows else alpha * sparse_row.score
            else:
                continue
            source_row = dense_row or sparse_row
            assert source_row is not None
            factor = _time_decay_factor(source_row.row, decay) if decay else None
            score = base_score * factor if factor is not None else base_score
            fused.append(
                _RankedRow(
                    row=source_row.row,
                    score=score,
                    origin_score=base_score,
                    time_score=factor,
                )
            )
        fused.sort(key=lambda item: (-item.score, str(item.row.id)))
        return fused

    def search_by_keywords(
        self,
        index_name: str,
        keywords: Optional[List[str]] = None,
        query: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        mode: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> SearchResult:
        self._validate_search(limit, offset, output_fields)
        self._validate_output_fields(output_fields)
        if limit == 0:
            return SearchResult()
        if mode not in (None, "bm25", "seekdb", "opengauss"):
            raise ValueError(f"unsupported SeekDB keyword mode {mode!r}")
        if fields and self.schema.content_field not in fields:
            raise SeekDBUnsupportedError(
                "SeekDB keyword search currently indexes the collection content field only"
            )
        if self.schema.content_field is None:
            raise SeekDBUnsupportedError("collection has no content text field")
        meta = self._require_index(index_name)
        internal = meta.get("_seekdb", {})
        if not internal.get("fulltext_enabled"):
            raise SeekDBUnsupportedError("SeekDB full-text index is not enabled")
        if keywords is not None and (
            not isinstance(keywords, list) or any(not isinstance(item, str) for item in keywords)
        ):
            raise TypeError("keywords must be a list of strings")
        text = (
            query if isinstance(query, str) and query.strip() else " ".join(keywords or []).strip()
        )
        if not text:
            return SearchResult()
        compiled = self._filter_compiler.compile(filters)
        match_expr = f"MATCH(r.`{self.schema.content_field}`) AGAINST (%s IN NATURAL LANGUAGE MODE)"
        clauses = [f"{match_expr} > 0"]
        params: list[Any] = [text]
        if compiled.sql != "TRUE":
            clauses.append(f"({compiled.sql})")
            params.extend(compiled.params)
        sql = (
            "SELECT r.`id` AS id, r.`metadata` AS metadata, r.`content` AS content, "
            f"r.`embedding` AS embedding, r.`sparse_embedding` AS sparse_embedding, "
            f"r.`sparse_terms` AS sparse_terms, {match_expr} AS _ov_score "
            f"FROM {_quote(self.physical_name)} r WHERE {' AND '.join(clauses)} "
            "ORDER BY _ov_score DESC, r.`id` ASC LIMIT %s OFFSET %s"
        )
        # SQL parameters are positional in textual order: the MATCH expression in
        # SELECT precedes the MATCH expression in WHERE.
        params = [text, *params, limit, offset]
        with self.client.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        result = SearchResult()
        for row in rows:
            decoded = _decode_row(row)
            result.data.append(
                SearchItemResult(
                    id=decoded.id,
                    fields=decoded.to_record(self.schema, output_fields),
                    score=float(row.get("_ov_score") or 0.0),
                )
            )
        return result

    def search_by_scalar(
        self,
        index_name: str,
        field: str,
        order: Optional[str] = "desc",
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        self._validate_search(limit, offset, output_fields)
        self._validate_output_fields(output_fields)
        if field not in self.schema.fields:
            raise ValueError(f"unknown SeekDB scalar-order field {field!r}")
        spec = self.schema.fields[field]
        if spec.field_type in {"vector", "sparse_vector"}:
            raise ValueError(f"field {field!r} cannot be used for scalar ordering")
        compiled = self._filter_compiler.compile(filters)
        direction = "DESC" if str(order or "desc").lower() == "desc" else "ASC"
        expr = self._filter_compiler._value_expr(field)
        where = f"WHERE {compiled.sql}" if compiled.sql != "TRUE" else ""
        sql = (
            "SELECT r.`id` AS id, r.`metadata` AS metadata, r.`content` AS content, "
            f"r.`embedding` AS embedding, r.`sparse_embedding` AS sparse_embedding, "
            f"r.`sparse_terms` AS sparse_terms FROM {_quote(self.physical_name)} r "
            f"{where} ORDER BY {expr} {direction}, r.`id` ASC LIMIT %s OFFSET %s"
        )
        with self.client.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(sql, (*compiled.params, limit, offset))
            rows = cursor.fetchall()
        result = SearchResult()
        for row in rows:
            decoded = _decode_row(row)
            result.data.append(
                SearchItemResult(
                    id=decoded.id,
                    fields=decoded.to_record(self.schema, output_fields),
                    score=0.0,
                )
            )
        return result

    def search_by_id(
        self,
        index_name: str,
        id: Any,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        source = self.fetch_data([id]).items
        if not source:
            return SearchResult()
        vector = source[0].fields.get(self.schema.vector_field)
        if not vector:
            return SearchResult()
        return self.search_by_vector(
            index_name,
            dense_vector=vector,
            limit=limit,
            offset=offset,
            filters=filters,
            output_fields=output_fields,
        )

    def search_by_random(
        self,
        index_name: str,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        advance: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        if advance:
            dim = self.schema.dimension
            vector = [random.uniform(-1.0, 1.0) for _ in range(dim)]
            return self.search_by_vector(
                index_name,
                dense_vector=vector,
                limit=limit,
                offset=offset,
                filters=filters,
                output_fields=output_fields,
                advance=advance,
            )
        self._validate_search(limit, offset, output_fields)
        self._validate_output_fields(output_fields)
        compiled = self._filter_compiler.compile(filters)
        where = f"WHERE {compiled.sql}" if compiled.sql != "TRUE" else ""
        sql = (
            "SELECT r.`id` AS id, r.`metadata` AS metadata, r.`content` AS content, "
            f"r.`embedding` AS embedding, r.`sparse_embedding` AS sparse_embedding, "
            f"r.`sparse_terms` AS sparse_terms FROM {_quote(self.physical_name)} r "
            f"{where} ORDER BY RAND() LIMIT %s OFFSET %s"
        )
        with self.client.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(sql, (*compiled.params, limit, offset))
            rows = cursor.fetchall()
        result = SearchResult()
        for row in rows:
            decoded = _decode_row(row)
            result.data.append(
                SearchItemResult(
                    id=decoded.id,
                    fields=decoded.to_record(self.schema, output_fields),
                    score=0.0,
                )
            )
        return result

    def search_by_multimodal(
        self,
        index_name: str,
        text: Optional[str],
        image: Optional[Any],
        video: Optional[Any],
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        del index_name, text, image, video, limit, offset, filters, output_fields
        raise SeekDBUnsupportedError("SeekDB adapter does not implement multimodal queries")

    def aggregate_data(
        self,
        index_name: str,
        op: str = "count",
        field: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        cond: Optional[Dict[str, Any]] = None,
    ) -> AggregateResult:
        del index_name
        if op != "count":
            raise SeekDBUnsupportedError(f"SeekDB aggregation supports count only, got {op!r}")
        compiled = self._filter_compiler.compile(filters)
        where = f"WHERE {compiled.sql}" if compiled.sql != "TRUE" else ""
        if field is None:
            with self.client.transaction() as connection, connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT COUNT(*) AS total FROM {_quote(self.physical_name)} r {where}",
                    compiled.params,
                )
                row = cursor.fetchone()
            return AggregateResult(agg={"_total": int(row["total"])}, op="count", field=None)

        if field not in self.schema.fields:
            raise ValueError(f"unknown SeekDB aggregation field {field!r}")
        if self.schema.fields[field].field_type in {"vector", "sparse_vector", "text"}:
            raise ValueError(f"field {field!r} cannot be used for grouped count")
        if field in self.schema.array_fields:
            raise SeekDBUnsupportedError(
                "SeekDB grouped count over array fields is not implemented"
            )
        expr = self._filter_compiler._value_expr(field, string=True)
        sql = (
            f"SELECT {expr} AS group_key, COUNT(*) AS total "
            f"FROM {_quote(self.physical_name)} r {where} "
            "GROUP BY group_key ORDER BY group_key"
        )
        with self.client.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(sql, compiled.params)
            rows = cursor.fetchall()
        grouped: dict[Any, int] = {}
        for row in rows:
            key = row["group_key"]
            count = int(row["total"])
            if not _matches_post_aggregate(count, cond):
                continue
            grouped[key if key is not None else "null"] = count
        return AggregateResult(agg=grouped, op="count", field=field)


def _quote(identifier: str) -> str:
    validate_identifier(identifier)
    return f"`{identifier}`"


def _default_pq_subquantizers(dimension: int) -> int:
    for candidate in range(min(8, dimension), 0, -1):
        if dimension % candidate == 0:
            return candidate
    return 1


def _sql_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8")
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("SeekDB JSON column did not contain an object")
    return parsed


def _decode_vector(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8")
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in text.strip("[]").split(",") if item.strip()]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("SeekDB VECTOR column did not contain a vector")
    return [float(item) for item in parsed]


def _decode_row(row: Mapping[str, Any]) -> _DecodedRow:
    primary = row.get("id")
    if isinstance(primary, (bytes, bytearray)):
        primary = bytes(primary).decode("utf-8")
    return _DecodedRow(
        id=str(primary),
        metadata=_loads(row.get("metadata") or "{}"),
        content=(
            bytes(row["content"]).decode("utf-8")
            if isinstance(row.get("content"), (bytes, bytearray))
            else row.get("content")
        ),
        embedding=_decode_vector(row.get("embedding")),
        sparse_embedding=(
            bytes(row["sparse_embedding"]).decode("utf-8")
            if isinstance(row.get("sparse_embedding"), (bytes, bytearray))
            else row.get("sparse_embedding")
        ),
        sparse_terms=(_loads(row["sparse_terms"]) if row.get("sparse_terms") is not None else None),
    )


def _distance_to_similarity(metric: str, value: Any) -> float:
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(distance):
        return 0.0
    if metric == "cosine":
        return 1.0 - distance
    if metric == "inner_product":
        return distance
    return 1.0 / (1.0 + max(distance, 0.0))


def _public_index_meta(meta: Mapping[str, Any], schema: SchemaSpec) -> dict[str, Any]:
    result = {key: value for key, value in meta.items() if key not in {"_seekdb"}}
    if "VectorIndex" in result:
        vector = dict(result["VectorIndex"])
        vector["Dimension"] = schema.dimension
        result["VectorIndex"] = vector
    return result


def _parse_time_decay(
    advance: Optional[Mapping[str, Any]], schema: SchemaSpec
) -> Optional[_TimeDecaySpec]:
    if advance is None:
        return None
    if not isinstance(advance, Mapping) or set(advance) != {"time_decay"}:
        raise SeekDBUnsupportedError("SeekDB advanced ranking supports time_decay only")
    request = advance.get("time_decay")
    if not isinstance(request, Mapping):
        raise ValueError("time_decay must be an object")
    field = str(request.get("field", "updated_at"))
    if field not in schema.date_time_fields:
        raise ValueError("time_decay field must be a declared date_time field")
    compiled_keys = {"field", "origin_ms", "offset_ms", "scale_ms", "decay"}
    if set(request) == compiled_keys:
        return _TimeDecaySpec(
            field=field,
            origin_ms=float(request["origin_ms"]),
            offset_ms=float(request["offset_ms"]),
            scale_ms=float(request["scale_ms"]),
            decay=float(request["decay"]),
        )
    try:
        from openviking.utils.time_decay import build_time_decay_fusion_spec
        from openviking.utils.time_utils import parse_iso_datetime
    except ImportError as exc:
        raise SeekDBUnsupportedError(
            "raw time_decay protection requires the OpenViking runtime"
        ) from exc
    if "protection" not in request:
        raise ValueError("time_decay requires protection")
    origin = request.get("origin")
    if isinstance(origin, str):
        origin = parse_iso_datetime(origin)
    elif origin is None:
        origin = datetime.now(timezone.utc)
    elif not isinstance(origin, datetime):
        raise ValueError("time_decay.origin must be an ISO-8601 string or datetime")
    spec = build_time_decay_fusion_spec(
        protection=request["protection"], origin=origin, field=field
    )
    return _TimeDecaySpec(
        field=spec.field,
        origin_ms=float(spec.origin_ms),
        offset_ms=float(spec.offset_ms),
        scale_ms=float(spec.scale_ms),
        decay=float(spec.decay),
    )


def _time_decay_factor(row: _DecodedRow, spec: _TimeDecaySpec) -> float:
    value = row.metadata.get(spec.field)
    if value is None or isinstance(value, bool):
        return 1.0
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return 1.0
    elapsed = max(0.0, abs(spec.origin_ms - timestamp) - spec.offset_ms)
    return math.exp(math.log(spec.decay) * elapsed / spec.scale_ms)


def _matches_post_aggregate(value: int, cond: Optional[Mapping[str, Any]]) -> bool:
    if not cond:
        return True
    for key, threshold in cond.items():
        if key == "gt" and not value > threshold:
            return False
        if key == "gte" and not value >= threshold:
            return False
        if key == "lt" and not value < threshold:
            return False
        if key == "lte" and not value <= threshold:
            return False
    return True
