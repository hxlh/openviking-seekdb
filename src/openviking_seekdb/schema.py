"""OpenViking collection-schema normalization for SeekDB."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from .errors import SeekDBConfigurationError

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_SUPPORTED_FIELD_TYPES = {
    "string",
    "path",
    "int32",
    "int64",
    "float",
    "float32",
    "double",
    "bool",
    "boolean",
    "date_time",
    "list<string>",
    "list<int64>",
    "text",
    "vector",
    "sparse_vector",
}
_TYPE_ALIASES = {
    "integer": "int64",
    "long": "int64",
    "double": "float",
    "boolean": "bool",
}


def validate_identifier(value: str, *, kind: str = "identifier") -> str:
    """Validate a SQL/JSON field identifier before it reaches string interpolation."""
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise SeekDBConfigurationError(
            f"Invalid {kind} {value!r}; use 1-64 ASCII letters, digits, or underscores, "
            "and do not start with a digit"
        )
    return value


def derive_physical_name(logical_name: str, prefix: str = "ov_", *, kind: str = "table") -> str:
    """Create a deterministic, SQL-safe physical name while preserving readability."""
    digest = hashlib.sha1(logical_name.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", logical_name).strip("_").lower() or "collection"
    safe_prefix = re.sub(r"[^A-Za-z0-9_]+", "_", prefix).strip("_").lower()
    if not safe_prefix:
        safe_prefix = "ov"
    max_len = 64
    suffix = f"_{digest}"
    head_len = max(1, max_len - len(safe_prefix) - len(suffix) - 1)
    candidate = f"{safe_prefix}_{slug[:head_len]}{suffix}"
    return validate_identifier(candidate, kind=f"physical {kind} name")


def derive_index_name(logical_collection: str, logical_index: str, role: str) -> str:
    digest = hashlib.sha1(
        f"{logical_collection}\0{logical_index}\0{role}".encode("utf-8")
    ).hexdigest()[:12]
    return validate_identifier(f"ov_{role[:1]}_{digest}", kind="physical index name")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    field_type: str
    is_primary_key: bool = False
    dimension: int = 0
    default_value: Any = None
    raw: Mapping[str, Any] = None  # type: ignore[assignment]

    @property
    def stored_in_json(self) -> bool:
        return not self.is_primary_key and self.field_type not in {
            "vector",
            "sparse_vector",
            "text",
        }

    @property
    def json_path(self) -> str:
        return f"$.{self.name}"


@dataclass(frozen=True)
class SchemaSpec:
    raw: Dict[str, Any]
    fields: Mapping[str, FieldSpec]
    primary_key: str
    vector_field: str
    sparse_vector_field: Optional[str]
    content_field: Optional[str]
    scalar_index: tuple[str, ...]
    path_fields: frozenset[str]
    array_fields: frozenset[str]
    date_time_fields: frozenset[str]

    @classmethod
    def from_mapping(cls, schema: Mapping[str, Any]) -> "SchemaSpec":
        if not isinstance(schema, Mapping):
            raise SeekDBConfigurationError("collection schema must be an object")
        normalized_raw = copy.deepcopy(dict(schema))
        raw_fields = normalized_raw.get("Fields")
        if not isinstance(raw_fields, list) or not raw_fields:
            raise SeekDBConfigurationError("collection schema must contain a non-empty Fields list")

        fields: dict[str, FieldSpec] = {}
        primary_key: Optional[str] = None
        vector_field: Optional[str] = None
        sparse_vector_field: Optional[str] = None
        content_field: Optional[str] = None
        for item in raw_fields:
            if not isinstance(item, Mapping):
                raise SeekDBConfigurationError("each schema field must be an object")
            field = copy.deepcopy(dict(item))
            name = validate_identifier(str(field.get("FieldName", "")), kind="field name")
            if name in fields:
                raise SeekDBConfigurationError(f"duplicate schema field {name!r}")
            raw_type = str(field.get("FieldType", "string")).strip().lower()
            field_type = _TYPE_ALIASES.get(raw_type, raw_type)
            if field_type not in _SUPPORTED_FIELD_TYPES:
                raise SeekDBConfigurationError(
                    f"unsupported SeekDB field type {raw_type!r} for field {name!r}"
                )
            dimension = 0
            if field_type == "vector":
                try:
                    dimension = int(field.get("Dim", field.get("Dimension", field.get("dim", 0))))
                except (TypeError, ValueError) as exc:
                    raise SeekDBConfigurationError(
                        f"vector field {name!r} has invalid Dim"
                    ) from exc
                if dimension <= 0:
                    raise SeekDBConfigurationError(f"vector field {name!r} requires positive Dim")
                if vector_field is not None:
                    raise SeekDBConfigurationError(
                        "SeekDB plugin supports exactly one vector field"
                    )
                vector_field = name
            elif field_type == "sparse_vector":
                if sparse_vector_field is not None:
                    raise SeekDBConfigurationError(
                        "SeekDB plugin supports at most one sparse_vector field"
                    )
                sparse_vector_field = name
            elif field_type == "text" and name == "content":
                content_field = name

            is_pk = bool(field.get("IsPrimaryKey", False))
            if is_pk:
                if primary_key is not None:
                    raise SeekDBConfigurationError("collection schema has multiple primary keys")
                if field_type not in {"string", "path"}:
                    raise SeekDBConfigurationError("SeekDB primary key must be a string field")
                primary_key = name
            fields[name] = FieldSpec(
                name=name,
                field_type=field_type,
                is_primary_key=is_pk,
                dimension=dimension,
                default_value=field.get("DefaultValue"),
                raw=field,
            )

        if primary_key is None:
            if "id" not in fields:
                raise SeekDBConfigurationError(
                    "collection schema requires an 'id' primary-key field"
                )
            item = fields["id"]
            fields["id"] = FieldSpec(
                name=item.name,
                field_type=item.field_type,
                is_primary_key=True,
                dimension=item.dimension,
                default_value=item.default_value,
                raw={**dict(item.raw), "IsPrimaryKey": True},
            )
            primary_key = "id"
            for index, item in enumerate(normalized_raw["Fields"]):
                if item.get("FieldName") == primary_key:
                    normalized_raw["Fields"][index] = dict(item, IsPrimaryKey=True)
                    break
        if primary_key != "id":
            # The rest of OpenViking assumes a conventional ``id`` key.
            raise SeekDBConfigurationError(
                "OpenViking SeekDB collections must use 'id' as primary key"
            )
        if vector_field is None:
            raise SeekDBConfigurationError("collection schema requires one vector field")
        if content_field is not None and fields[content_field].field_type != "text":
            raise SeekDBConfigurationError("content field must have FieldType='text'")

        scalar_index = normalized_raw.get("ScalarIndex", [])
        if scalar_index is None:
            scalar_index = []
        if not isinstance(scalar_index, list):
            raise SeekDBConfigurationError("ScalarIndex must be a list")
        scalar_names = []
        for name in scalar_index:
            name = validate_identifier(str(name), kind="scalar-index field name")
            if name not in fields:
                raise SeekDBConfigurationError(f"ScalarIndex refers to unknown field {name!r}")
            if name not in scalar_names:
                scalar_names.append(name)

        return cls(
            raw=normalized_raw,
            fields=fields,
            primary_key=primary_key,
            vector_field=vector_field,
            sparse_vector_field=sparse_vector_field,
            content_field=content_field,
            scalar_index=tuple(scalar_names),
            path_fields=frozenset(
                name for name, spec in fields.items() if spec.field_type == "path"
            ),
            array_fields=frozenset(
                name
                for name, spec in fields.items()
                if spec.field_type in {"list<string>", "list<int64>"}
            ),
            date_time_fields=frozenset(
                name for name, spec in fields.items() if spec.field_type == "date_time"
            ),
        )

    @property
    def dimension(self) -> int:
        return self.fields[self.vector_field].dimension

    @property
    def collection_name(self) -> str:
        value = self.raw.get("CollectionName", "")
        return str(value)

    @property
    def description(self) -> str:
        return str(self.raw.get("Description", ""))

    @property
    def metadata_field_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, spec in self.fields.items()
            if name != self.primary_key
            and spec.field_type not in {"vector", "sparse_vector", "text"}
        )

    @property
    def all_output_fields(self) -> tuple[str, ...]:
        return tuple(self.fields)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)

    def with_updates(
        self,
        *,
        new_fields: Optional[Iterable[Mapping[str, Any]]] = None,
        description: Optional[str] = None,
        scalar_index: Optional[Iterable[str]] = None,
    ) -> "SchemaSpec":
        raw = copy.deepcopy(self.raw)
        if new_fields is not None:
            existing = list(raw.get("Fields", []))
            by_name = {item.get("FieldName"): item for item in existing}
            for field in new_fields:
                candidate = copy.deepcopy(dict(field))
                name = validate_identifier(str(candidate.get("FieldName", "")), kind="field name")
                if name in by_name:
                    by_name[name].update(candidate)
                else:
                    existing.append(candidate)
                    by_name[name] = candidate
            raw["Fields"] = existing
        if description is not None:
            raw["Description"] = description
        if scalar_index is not None:
            raw["ScalarIndex"] = list(dict.fromkeys(str(item) for item in scalar_index))
        updated = self.from_mapping(raw)
        if updated.dimension != self.dimension:
            raise SeekDBConfigurationError(
                "changing the dense vector dimension requires recreating the collection"
            )
        if updated.vector_field != self.vector_field:
            raise SeekDBConfigurationError("changing the vector field name is not supported")
        return updated

    def encode_metadata(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Validate and normalize scalar fields for the collection's JSON column."""
        normalized: Dict[str, Any] = {}
        for name, value in record.items():
            if name in {"id", self.primary_key, self.vector_field, self.sparse_vector_field}:
                continue
            spec = self.fields.get(name)
            if spec is None:
                raise ValueError(f"unknown SeekDB field {name!r}")
            if name == self.content_field:
                continue
            if value is None:
                continue
            normalized[name] = normalize_field_value(spec, value)
        json.dumps(normalized, ensure_ascii=False, allow_nan=False)
        return normalized

    def content_value(self, record: Mapping[str, Any]) -> Any:
        if not self.content_field:
            return None
        value = record.get(self.content_field)
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        return value

    def sparse_original(self, record: Mapping[str, Any]) -> Optional[Dict[str, float]]:
        if not self.sparse_vector_field:
            return None
        value = record.get(self.sparse_vector_field)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError(f"{self.sparse_vector_field} must be an object")
        return {str(key): float(weight) for key, weight in value.items()}

    def vector_value(self, record: Mapping[str, Any]) -> Optional[list[float]]:
        value = record.get(self.vector_field)
        if value is None:
            return None
        return normalize_vector(value, self.dimension, field_name=self.vector_field)


def normalize_vector(value: Any, dimension: int, *, field_name: str = "vector") -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple")
    if dimension > 0 and len(value) != dimension:
        raise ValueError(f"{field_name} dimension mismatch: expected {dimension}, got {len(value)}")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{field_name}[{index}] must be numeric")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{index}] must be numeric") from exc
        if number != number or number in {float("inf"), float("-inf")}:
            raise ValueError(f"{field_name}[{index}] must be finite")
        result.append(number)
    return result


def normalize_field_value(spec: FieldSpec, value: Any) -> Any:
    field_type = spec.field_type
    if field_type in {"string", "path", "text"}:
        return value if isinstance(value, str) else str(value)
    if field_type == "date_time":
        return datetime_to_epoch_ms(value)
    if field_type == "bool":
        if isinstance(value, bool):
            return value
        if value in {0, 1}:
            return bool(value)
        raise ValueError(f"{spec.name} must be boolean")
    if field_type in {"int32", "int64"}:
        if isinstance(value, bool):
            raise ValueError(f"{spec.name} must be an integer")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{spec.name} must be an integer") from exc
        if isinstance(value, float) and value != number:
            raise ValueError(f"{spec.name} must be an integer")
        return number
    if field_type in {"float", "float32"}:
        if isinstance(value, bool):
            raise ValueError(f"{spec.name} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{spec.name} must be numeric") from exc
        if number != number or number in {float("inf"), float("-inf")}:
            raise ValueError(f"{spec.name} must be finite")
        return number
    if field_type == "list<string>":
        if isinstance(value, str):
            value = value.split(";")
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{spec.name} must be a list of strings")
        return [item if isinstance(item, str) else str(item) for item in value]
    if field_type == "list<int64>":
        if isinstance(value, str):
            value = value.split(";")
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{spec.name} must be a list of integers")
        result = []
        for item in value:
            if isinstance(item, bool):
                raise ValueError(f"{spec.name} must contain integers")
            try:
                result.append(int(item))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{spec.name} must contain integers") from exc
        return result
    raise ValueError(f"field {spec.name!r} with type {field_type!r} is not scalar")


def datetime_to_epoch_ms(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("date_time value cannot be boolean")
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        raise ValueError("date_time value must be an ISO-8601 string or epoch milliseconds")
    raw = value.strip()
    if not raw:
        raise ValueError("date_time value cannot be empty")
    try:
        return int(raw)
    except ValueError:
        pass
    from datetime import datetime, timezone

    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def derive_internal_tables(metadata_prefix: str) -> tuple[str, str]:
    collections = validate_identifier(f"{metadata_prefix}_collections", kind="metadata table name")
    indexes = validate_identifier(f"{metadata_prefix}_indexes", kind="metadata table name")
    return collections, indexes
