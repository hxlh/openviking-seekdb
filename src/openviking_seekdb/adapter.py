"""OpenViking CollectionAdapter entrypoint for OceanBase SeekDB."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from openviking.storage.vectordb.collection.collection import Collection
from openviking.storage.vectordb_adapters.base import CollectionAdapter

from .client import SeekDBClient
from .collection import SeekDBCollection
from .config import SeekDBConfig
from .errors import SeekDBCollectionMissingError, SeekDBConfigurationError
from .schema import SchemaSpec, derive_physical_name

logger = logging.getLogger(__name__)


class SeekDBCollectionAdapter(CollectionAdapter):
    """OpenViking v0.5.3 adapter backed by SeekDB server mode.

    OpenViking v0.5.3 hard-codes keyword/full-text support to
    ``volcengine``, ``vikingdb`` and ``opengauss``. The adapter therefore reports
    the ``opengauss`` compatibility mode by default while retaining
    ``backend_name = "seekdb"``. This lets SearchService and grep use SeekDB's
    native full-text capability without patching OpenViking internals.
    """

    backend_name = "seekdb"
    _DATA_BATCH_SIZE = 100

    def __init__(
        self,
        *,
        collection_name: str,
        seekdb_config: SeekDBConfig,
        index_name: str = "default",
        distance_metric: str = "cosine",
        dimension: int = 0,
        sparse_weight: float = 0.0,
        client: Optional[SeekDBClient] = None,
    ) -> None:
        super().__init__(collection_name=collection_name, index_name=index_name)
        self.config = seekdb_config
        self._distance_metric = (distance_metric or "cosine").strip().lower()
        if self._distance_metric == "ip":
            self._distance_metric = "inner_product"
        if self._distance_metric not in {"cosine", "l2", "inner_product"}:
            raise SeekDBConfigurationError(
                "distance_metric must be one of cosine, l2, ip, or inner_product"
            )
        self._dimension = int(dimension)
        self._sparse_weight = float(sparse_weight)
        self._vector_index_type = seekdb_config.vector_index_type
        self._client = client or SeekDBClient(seekdb_config)
        self._load_lock = threading.RLock()
        self.mode = (
            seekdb_config.compatibility_mode
            if seekdb_config.fulltext and seekdb_config.openviking_fulltext_compat
            else "seekdb"
        )
        self.USE_CONTENT_FIELD = bool(seekdb_config.fulltext)
        if self._sparse_weight < 0 or self._sparse_weight > 1:
            raise SeekDBConfigurationError("sparse_weight must be in [0, 1]")

    @classmethod
    def from_config(cls, config: Any) -> "SeekDBCollectionAdapter":
        custom_params = getattr(config, "custom_params", {}) or {}
        seekdb_config = SeekDBConfig.from_custom_params(custom_params)
        return cls(
            collection_name=getattr(config, "name", None) or "context",
            seekdb_config=seekdb_config,
            index_name=getattr(config, "index_name", None) or "default",
            distance_metric=getattr(config, "distance_metric", None) or "cosine",
            dimension=int(getattr(config, "dimension", 0) or 0),
            sparse_weight=float(getattr(config, "sparse_weight", 0.0) or 0.0),
        )

    def _load_existing_collection_if_needed(self) -> None:
        with self._load_lock:
            if self._collection is not None:
                return
            record = self._client.load_collection(self._collection_name)
            if record is None:
                return
            physical_name = str(record["physical_name"])
            if not self._client.table_exists(physical_name):
                raise SeekDBCollectionMissingError(
                    f"SeekDB catalog contains collection {self._collection_name!r}, but physical "
                    f"table {physical_name!r} is missing"
                )
            schema = SchemaSpec.from_mapping(record["schema"])
            if self._dimension and schema.dimension != self._dimension:
                raise SeekDBConfigurationError(
                    f"OpenViking vector dimension {self._dimension} does not match SeekDB "
                    f"collection dimension {schema.dimension}"
                )
            self._dimension = schema.dimension
            index_meta = self._client.load_index(self._collection_name, self._index_name)
            if index_meta is None:
                raise SeekDBCollectionMissingError(
                    f"SeekDB collection {self._collection_name!r} has no logical index "
                    f"{self._index_name!r}"
                )
            self._vector_index_type = str(
                index_meta.get("_seekdb", {}).get(
                    "vector_index_type",
                    index_meta.get("VectorIndex", {}).get("IndexType", "hnsw"),
                )
            )
            seekdb_collection = SeekDBCollection(
                client=self._client,
                config=self.config,
                logical_name=self._collection_name,
                physical_name=physical_name,
                schema=schema,
                sparse_weight=self._sparse_weight,
                index_meta=index_meta,
            )
            self._collection = Collection(seekdb_collection)

    def _create_backend_collection(self, meta: Dict[str, Any]) -> Collection:
        schema = SchemaSpec.from_mapping(meta)
        if self._dimension and schema.dimension != self._dimension:
            raise SeekDBConfigurationError(
                f"input vector dimension {self._dimension} does not match schema dimension "
                f"{schema.dimension}"
            )
        self._dimension = schema.dimension
        physical_name = derive_physical_name(self._collection_name, self.config.table_prefix)
        seekdb_collection = SeekDBCollection(
            client=self._client,
            config=self.config,
            logical_name=self._collection_name,
            physical_name=physical_name,
            schema=schema,
            sparse_weight=self._sparse_weight,
        )
        seekdb_collection.create_physical_table()
        try:
            self._client.save_collection(
                self._collection_name,
                physical_name,
                schema.to_dict(),
            )
        except Exception:
            seekdb_collection.drop_physical_table()
            raise
        return Collection(seekdb_collection)

    def create_collection(
        self,
        name: str,
        schema: Dict[str, Any],
        *,
        distance: str,
        sparse_weight: float,
        index_name: str,
    ) -> bool:
        self._sparse_weight = float(sparse_weight)
        try:
            return super().create_collection(
                name,
                schema,
                distance=distance,
                sparse_weight=sparse_weight,
                index_name=index_name,
            )
        except Exception:
            if self._collection is not None:
                try:
                    self._collection.drop()
                except Exception:
                    logger.exception(
                        "Failed to remove partially created SeekDB collection %s", name
                    )
                self._collection = None
            raise

    def update_data(self, data_list: List[Dict[str, Any]]) -> list[str]:
        normalized = [self._normalize_record_for_write(record) for record in data_list]
        self.begin_bulk_ingest()
        try:
            result = self.get_collection().update_data(normalized)
        finally:
            self.end_bulk_ingest()
        return list(getattr(result, "ids", result) or [])

    def upsert(self, data):
        self.begin_bulk_ingest()
        try:
            return super().upsert(data)
        finally:
            self.end_bulk_ingest()

    def delete(self, *, ids=None, filter=None):
        self.begin_bulk_ingest()
        try:
            return super().delete(ids=ids, filter=filter)
        finally:
            self.end_bulk_ingest()

    def begin_bulk_ingest(self) -> None:
        if self._collection is not None:
            self._collection.begin_bulk_ingest()

    def end_bulk_ingest(self) -> None:
        if self._collection is not None:
            self._collection.end_bulk_ingest()

    def close(self) -> None:
        super().close()
        self._client.close()

    def probe_capabilities(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "openviking_mode": self.mode,
            "dense": True,
            "sparse": self._sparse_weight > 0,
            "full_text": bool(self.USE_CONTENT_FIELD),
            "hybrid": self._sparse_weight > 0,
            "partial_update": True,
            "distance_metric": self._distance_metric,
            "vector_index_type": self._vector_index_type,
        }
