"""Configuration parsing for the OpenViking SeekDB plugin."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .errors import SeekDBConfigurationError

_SUPPORTED_FULLTEXT_ANALYZERS = {"ik", "space", "beng", "ngram", "ngram2"}
_SUPPORTED_MODES = {"opengauss", "seekdb"}
_VECTOR_INDEX_TYPES = frozenset({"flat", "hnsw", "hnsw_sq", "hnsw_bq", "ivf_flat", "ivf_pq"})
_VECTOR_INDEX_ALIASES = {
    "ivf": "ivf_flat",
    "ivfflat": "ivf_flat",
    "ivf_pq": "ivf_pq",
    "ivfpq": "ivf_pq",
}
_HNSW_REFINE_TYPES = frozenset({"sq8", "fp32"})


def _as_int(payload: Mapping[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    raw = payload.get(key, default)
    if isinstance(raw, bool):
        raise SeekDBConfigurationError(f"custom_params.{key} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SeekDBConfigurationError(f"custom_params.{key} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise SeekDBConfigurationError(
            f"custom_params.{key} must be between {minimum} and {maximum}, got {value}"
        )
    return value


def _as_float(
    payload: Mapping[str, Any], key: str, default: float, minimum: float, maximum: float
) -> float:
    raw = payload.get(key, default)
    if isinstance(raw, bool):
        raise SeekDBConfigurationError(f"custom_params.{key} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SeekDBConfigurationError(f"custom_params.{key} must be a number") from exc
    if not minimum <= value <= maximum:
        raise SeekDBConfigurationError(
            f"custom_params.{key} must be between {minimum} and {maximum}, got {value}"
        )
    return value


def _as_bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = payload.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    raise SeekDBConfigurationError(f"custom_params.{key} must be a boolean")


def _as_optional_int(
    payload: Mapping[str, Any], key: str, minimum: int, maximum: int
) -> Optional[int]:
    if payload.get(key) is None:
        return None
    return _as_int(payload, key, minimum, minimum, maximum)


def _as_optional_float(
    payload: Mapping[str, Any], key: str, minimum: float, maximum: float
) -> Optional[float]:
    if payload.get(key) is None:
        return None
    return _as_float(payload, key, minimum, minimum, maximum)


def _normalize_vector_index_type(value: Any) -> str:
    normalized = str(value or "hnsw").strip().lower().replace("-", "_")
    normalized = _VECTOR_INDEX_ALIASES.get(normalized, normalized)
    if normalized not in _VECTOR_INDEX_TYPES:
        raise SeekDBConfigurationError(
            f"Unsupported custom_params.vector_index_type {normalized!r}; "
            f"choose from {sorted(_VECTOR_INDEX_TYPES)}"
        )
    return normalized


@dataclass(frozen=True)
class SeekDBConfig:
    """Connection, schema and retrieval options for SeekDB server mode."""

    host: str = "127.0.0.1"
    port: int = 2881
    database: str = "openviking"
    user: str = "root"
    tenant: Optional[str] = None
    password: str = ""
    password_env: str = "SEEKDB_PASSWORD"
    charset: str = "utf8mb4"
    connect_timeout: int = 10
    read_timeout: int = 60
    write_timeout: int = 60
    pool_size: int = 8
    connect_retries: int = 3
    metadata_prefix: str = "_ov_seekdb"
    table_prefix: str = "ov_"
    fulltext: bool = True
    fulltext_analyzer: str = "ik"
    fulltext_properties: Mapping[str, Any] = field(default_factory=dict)
    sparse_max_dimension: int = 500_000
    sparse_max_nonzero: int = 4_096
    sparse_prune: bool = False
    sparse_refine: bool = False
    vector_index_type: str = "hnsw"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    hnsw_extra_info_max_size: Optional[int] = None
    hnsw_refine_k: Optional[float] = None
    hnsw_refine_type: Optional[str] = None
    hnsw_bq_bits_query: Optional[int] = None
    hnsw_bq_use_fht: Optional[bool] = None
    ivf_nlist: int = 128
    ivf_sample_per_nlist: int = 256
    ivf_nbits: int = 8
    ivf_pq_m: Optional[int] = None
    ivf_nprobes: Optional[int] = None
    candidate_limit: int = 1_000
    refresh_after_write: bool = True
    openviking_fulltext_compat: bool = True
    compatibility_mode: str = "opengauss"
    ssl_ca: Optional[str] = None
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_verify_cert: bool = True
    ssl_verify_identity: bool = True

    @classmethod
    def from_custom_params(cls, custom_params: Mapping[str, Any] | None) -> "SeekDBConfig":
        source = dict(custom_params or {})
        nested = source.pop("seekdb", None)
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise SeekDBConfigurationError("custom_params.seekdb must be an object")
            source.update(nested)

        host = str(source.get("host", "127.0.0.1")).strip()
        database = str(source.get("database", "openviking")).strip()
        user = str(source.get("user", "root")).strip()
        tenant_raw = source.get("tenant")
        tenant = str(tenant_raw).strip() if tenant_raw else None
        charset = str(source.get("charset", "utf8mb4")).strip()
        if not host:
            raise SeekDBConfigurationError("custom_params.host cannot be empty")
        if not database:
            raise SeekDBConfigurationError("custom_params.database cannot be empty")
        if not user:
            raise SeekDBConfigurationError("custom_params.user cannot be empty")
        if charset not in {"utf8mb4", "utf8"}:
            raise SeekDBConfigurationError("custom_params.charset must be utf8mb4 or utf8")

        password_env = str(source.get("password_env", "SEEKDB_PASSWORD")).strip()
        direct_password = source.get("password")
        password = (
            str(direct_password)
            if direct_password is not None
            else (os.getenv(password_env, "") if password_env else "")
        )

        analyzer = str(source.get("fulltext_analyzer", "ik")).strip().lower()
        if analyzer not in _SUPPORTED_FULLTEXT_ANALYZERS:
            raise SeekDBConfigurationError(
                f"Unsupported SeekDB fulltext analyzer {analyzer!r}; "
                f"choose from {sorted(_SUPPORTED_FULLTEXT_ANALYZERS)}"
            )
        properties = source.get("fulltext_properties", {})
        if properties is None:
            properties = {}
        if not isinstance(properties, Mapping):
            raise SeekDBConfigurationError("custom_params.fulltext_properties must be an object")

        compatibility_mode = (
            str(
                source.get(
                    "compatibility_mode",
                    "opengauss"
                    if _as_bool(source, "openviking_fulltext_compat", True)
                    else "seekdb",
                )
            )
            .strip()
            .lower()
        )
        if compatibility_mode not in _SUPPORTED_MODES:
            raise SeekDBConfigurationError(
                "custom_params.compatibility_mode must be 'opengauss' or 'seekdb'"
            )

        metadata_prefix = str(source.get("metadata_prefix", "_ov_seekdb")).strip()
        table_prefix = str(source.get("table_prefix", "ov_")).strip()
        if not metadata_prefix or not table_prefix:
            raise SeekDBConfigurationError("SeekDB table prefixes cannot be empty")

        vector_index_type = _normalize_vector_index_type(
            source.get("vector_index_type", source.get("index_type", "hnsw"))
        )
        refine_type_raw = source.get("hnsw_refine_type")
        refine_type = str(refine_type_raw).strip().lower() if refine_type_raw is not None else None
        if refine_type is not None and refine_type not in _HNSW_REFINE_TYPES:
            raise SeekDBConfigurationError("custom_params.hnsw_refine_type must be 'sq8' or 'fp32'")
        hnsw_bq_use_fht_raw = source.get("hnsw_bq_use_fht")
        hnsw_bq_use_fht = (
            _as_bool(source, "hnsw_bq_use_fht", False) if hnsw_bq_use_fht_raw is not None else None
        )
        hnsw_bq_bits_query = _as_optional_int(source, "hnsw_bq_bits_query", 0, 32)
        if hnsw_bq_bits_query not in {None, 0, 4, 32}:
            raise SeekDBConfigurationError(
                "custom_params.hnsw_bq_bits_query must be one of 0, 4, or 32"
            )

        config = cls(
            host=host,
            port=_as_int(source, "port", 2881, 1, 65535),
            database=database,
            user=user,
            tenant=tenant,
            password=password,
            password_env=password_env,
            charset=charset,
            connect_timeout=_as_int(source, "connect_timeout", 10, 1, 600),
            read_timeout=_as_int(source, "read_timeout", 60, 1, 3600),
            write_timeout=_as_int(source, "write_timeout", 60, 1, 3600),
            pool_size=_as_int(source, "pool_size", 8, 1, 128),
            connect_retries=_as_int(source, "connect_retries", 3, 1, 20),
            metadata_prefix=metadata_prefix,
            table_prefix=table_prefix,
            fulltext=_as_bool(source, "fulltext", True),
            fulltext_analyzer=analyzer,
            fulltext_properties=dict(properties),
            sparse_max_dimension=_as_int(source, "sparse_max_dimension", 500_000, 1, 500_000),
            sparse_max_nonzero=_as_int(source, "sparse_max_nonzero", 4_096, 1, 65_535),
            sparse_prune=_as_bool(source, "sparse_prune", False),
            sparse_refine=_as_bool(source, "sparse_refine", False),
            vector_index_type=vector_index_type,
            hnsw_m=_as_int(source, "hnsw_m", 16, 5, 128),
            hnsw_ef_construction=_as_int(source, "hnsw_ef_construction", 200, 5, 1000),
            hnsw_ef_search=_as_int(source, "hnsw_ef_search", 64, 1, 1000),
            hnsw_extra_info_max_size=_as_optional_int(
                source, "hnsw_extra_info_max_size", 0, 16_384
            ),
            hnsw_refine_k=_as_optional_float(source, "hnsw_refine_k", 1.0, 1_000.0),
            hnsw_refine_type=refine_type,
            hnsw_bq_bits_query=hnsw_bq_bits_query,
            hnsw_bq_use_fht=hnsw_bq_use_fht,
            ivf_nlist=_as_int(source, "ivf_nlist", 128, 1, 65_536),
            ivf_sample_per_nlist=_as_int(source, "ivf_sample_per_nlist", 256, 1, 2_147_483_647),
            ivf_nbits=_as_int(source, "ivf_nbits", 8, 1, 24),
            ivf_pq_m=_as_optional_int(source, "ivf_pq_m", 1, 65_536),
            ivf_nprobes=_as_optional_int(source, "ivf_nprobes", 1, 65_536),
            candidate_limit=_as_int(source, "candidate_limit", 1000, 1, 16384),
            refresh_after_write=_as_bool(source, "refresh_after_write", True),
            openviking_fulltext_compat=_as_bool(source, "openviking_fulltext_compat", True),
            compatibility_mode=compatibility_mode,
            ssl_ca=_optional_str(source.get("ssl_ca")),
            ssl_cert=_optional_str(source.get("ssl_cert")),
            ssl_key=_optional_str(source.get("ssl_key")),
            ssl_verify_cert=_as_bool(source, "ssl_verify_cert", True),
            ssl_verify_identity=_as_bool(source, "ssl_verify_identity", True),
        )
        if config.vector_index_type.startswith("hnsw") and (
            config.hnsw_ef_construction <= config.hnsw_m
        ):
            raise SeekDBConfigurationError("hnsw_ef_construction must be greater than hnsw_m")
        bq_options = {
            "hnsw_refine_k": config.hnsw_refine_k,
            "hnsw_refine_type": config.hnsw_refine_type,
            "hnsw_bq_bits_query": config.hnsw_bq_bits_query,
            "hnsw_bq_use_fht": config.hnsw_bq_use_fht,
        }
        if config.vector_index_type != "hnsw_bq" and any(
            value is not None for value in bq_options.values()
        ):
            raise SeekDBConfigurationError(
                "HNSW_BQ parameters require custom_params.vector_index_type='hnsw_bq'"
            )
        if config.ivf_nprobes is not None and config.ivf_nprobes > config.ivf_nlist:
            raise SeekDBConfigurationError("ivf_nprobes cannot exceed ivf_nlist")
        if config.sparse_refine and not config.sparse_prune:
            raise SeekDBConfigurationError("sparse_refine requires sparse_prune=true")
        return config

    def connection_user(self) -> str:
        if not self.tenant or "@" in self.user:
            return self.user
        return f"{self.user}@{self.tenant}"

    def ssl_options(self) -> Optional[dict[str, Any]]:
        values = {
            "ca": self.ssl_ca,
            "cert": self.ssl_cert,
            "key": self.ssl_key,
        }
        if not any(values.values()):
            return None
        return {
            "ca": self.ssl_ca,
            "cert": self.ssl_cert,
            "key": self.ssl_key,
            "check_hostname": self.ssl_verify_identity,
        }


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
