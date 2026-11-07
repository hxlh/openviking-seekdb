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


def test_keyword_search_binds_match_and_scope_filters_in_sql_order(schema_dict):
    from contextlib import contextmanager

    from openviking_seekdb.collection import SeekDBCollection
    from openviking_seekdb.config import SeekDBConfig
    from openviking_seekdb.filters import SeekDBFilterCompiler
    from openviking_seekdb.schema import SchemaSpec

    captured = {}

    class Cursor:
        def execute(self, sql, params):
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class Connection:
        def cursor(self):
            return Cursor()

    class Client:
        @contextmanager
        def transaction(self):
            yield Connection()

        def table_indexes(self, _physical_name):
            return {"ov_f_test"}

    collection = object.__new__(SeekDBCollection)
    collection.client = Client()
    collection.config = SeekDBConfig.from_custom_params({"fulltext": True})
    collection.logical_name = "context"
    collection.physical_name = "ov_context_test"
    collection.schema = SchemaSpec.from_mapping(schema_dict)
    collection._indexes = {
        "default": {
            "_seekdb": {
                "vector_index_type": "flat",
                "sparse_enabled": False,
                "fulltext_enabled": True,
                "fulltext_index": "ov_f_test",
            }
        }
    }
    collection._filter_compiler = SeekDBFilterCompiler(collection.schema)

    result = collection.search_by_keywords(
        "default",
        query="machine learning",
        limit=3,
        filters={"op": "must", "field": "account_id", "conds": ["acct-a"]},
        output_fields=["id"],
    )

    assert result.data == []
    assert captured["sql"].count("%s") == 5
    assert tuple(captured["params"]) == (
        "machine learning",
        "machine learning",
        '["acct-a"]',
        3,
        0,
    )
