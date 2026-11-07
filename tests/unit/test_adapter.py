from openviking.storage.vectordb_adapters.factory import create_collection_adapter
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig

from openviking_seekdb.adapter import SeekDBCollectionAdapter


def test_dynamic_factory_loading_is_lazy_and_reports_compatibility_mode():
    config = VectorDBBackendConfig(
        backend="openviking_seekdb.adapter.SeekDBCollectionAdapter",
        name="context",
        index_name="default",
        dimension=4,
        sparse_weight=0.2,
        custom_params={
            "host": "127.0.0.1",
            "database": "openviking",
            "fulltext": True,
        },
    )
    adapter = create_collection_adapter(config)
    assert isinstance(adapter, SeekDBCollectionAdapter)
    assert adapter.backend_name == "seekdb"
    assert adapter.mode == "opengauss"
    assert adapter.USE_CONTENT_FIELD is True
    capabilities = adapter.probe_capabilities()
    assert capabilities["full_text"] is True
    assert capabilities["hybrid"] is True
    assert capabilities["vector_index_type"] == "hnsw"
