import pytest

from openviking_seekdb.config import SeekDBConfig
from openviking_seekdb.errors import SeekDBConfigurationError


def test_password_env_and_nested_options(monkeypatch):
    monkeypatch.setenv("CUSTOM_SEEKDB_SECRET", "secret-value")
    config = SeekDBConfig.from_custom_params(
        {
            "host": "outer",
            "seekdb": {
                "host": "seekdb.internal",
                "port": "2882",
                "database": "ov_test",
                "user": "app",
                "tenant": "tenant_a",
                "password_env": "CUSTOM_SEEKDB_SECRET",
                "pool_size": 4,
                "fulltext_analyzer": "space",
            },
        }
    )
    assert config.host == "seekdb.internal"
    assert config.port == 2882
    assert config.password == "secret-value"
    assert config.connection_user() == "app@tenant_a"
    assert config.pool_size == 4
    assert config.fulltext_analyzer == "space"


def test_explicit_password_takes_precedence(monkeypatch):
    monkeypatch.setenv("SEEKDB_PASSWORD", "from-env")
    config = SeekDBConfig.from_custom_params({"password": "explicit"})
    assert config.password == "explicit"


def test_fulltext_compatibility_mode_defaults_to_opengauss():
    config = SeekDBConfig.from_custom_params({})
    assert config.compatibility_mode == "opengauss"
    assert config.openviking_fulltext_compat is True


def test_disable_compatibility_uses_native_mode():
    config = SeekDBConfig.from_custom_params({"openviking_fulltext_compat": False})
    assert config.compatibility_mode == "seekdb"


def test_refresh_after_write_can_be_disabled():
    config = SeekDBConfig.from_custom_params({"refresh_after_write": False})
    assert config.refresh_after_write is False


def test_vector_index_defaults_to_hnsw():
    config = SeekDBConfig.from_custom_params({})
    assert config.vector_index_type == "hnsw"


@pytest.mark.parametrize(
    ("configured", "normalized"),
    [
        ("HNSW-SQ", "hnsw_sq"),
        ("hnsw_bq", "hnsw_bq"),
        ("ivf", "ivf_flat"),
        ("ivf-pq", "ivf_pq"),
        ("FLAT", "flat"),
    ],
)
def test_vector_index_type_aliases(configured, normalized):
    config = SeekDBConfig.from_custom_params({"vector_index_type": configured})
    assert config.vector_index_type == normalized


def test_hnsw_bq_and_ivf_options_are_normalized():
    bq = SeekDBConfig.from_custom_params(
        {
            "vector_index_type": "hnsw_bq",
            "hnsw_refine_k": "2.5",
            "hnsw_refine_type": "FP32",
            "hnsw_bq_bits_query": "4",
            "hnsw_bq_use_fht": "false",
        }
    )
    assert bq.hnsw_refine_k == 2.5
    assert bq.hnsw_refine_type == "fp32"
    assert bq.hnsw_bq_bits_query == 4
    assert bq.hnsw_bq_use_fht is False

    ivf = SeekDBConfig.from_custom_params(
        {
            "vector_index_type": "ivf_pq",
            "ivf_nlist": 16,
            "ivf_sample_per_nlist": 64,
            "ivf_nbits": 10,
            "ivf_pq_m": 8,
            "ivf_nprobes": 4,
        }
    )
    assert ivf.ivf_nlist == 16
    assert ivf.ivf_sample_per_nlist == 64
    assert ivf.ivf_nbits == 10
    assert ivf.ivf_pq_m == 8
    assert ivf.ivf_nprobes == 4


def test_bq_options_require_hnsw_bq():
    with pytest.raises(SeekDBConfigurationError, match="HNSW_BQ parameters"):
        SeekDBConfig.from_custom_params({"vector_index_type": "hnsw", "hnsw_refine_k": 2.0})


def test_ivf_nprobes_cannot_exceed_nlist():
    with pytest.raises(SeekDBConfigurationError, match="cannot exceed"):
        SeekDBConfig.from_custom_params(
            {"vector_index_type": "ivf_flat", "ivf_nlist": 4, "ivf_nprobes": 8}
        )
