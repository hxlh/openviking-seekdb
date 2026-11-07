from __future__ import annotations

import os
import uuid

import pymysql
import pytest
from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.expr import And, Contains, Eq, In, PathScope, Range, RawDSL
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig

from openviking_seekdb.adapter import SeekDBCollectionAdapter
from openviking_seekdb.schema import derive_index_name

pytestmark = pytest.mark.integration

if not os.getenv("SEEKDB_TEST_DATABASE"):
    pytest.skip(
        "set SEEKDB_TEST_DATABASE (and optionally SEEKDB_TEST_HOST/PORT/USER/PASSWORD)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def adapter():
    host = os.getenv("SEEKDB_TEST_HOST", "127.0.0.1")
    port = int(os.getenv("SEEKDB_TEST_PORT", "2881"))
    database = os.environ["SEEKDB_TEST_DATABASE"]
    user = os.getenv("SEEKDB_TEST_USER", "root")
    password = os.getenv("SEEKDB_TEST_PASSWORD", "")
    collection_name = f"ov_it_{uuid.uuid4().hex[:8]}"

    config = VectorDBBackendConfig(
        backend="openviking_seekdb.adapter.SeekDBCollectionAdapter",
        name=collection_name,
        index_name="default",
        dimension=4,
        distance_metric="cosine",
        sparse_weight=0.4,
        custom_params={
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
            "pool_size": 2,
            "fulltext": True,
            "fulltext_analyzer": "ik",
            "candidate_limit": 32,
        },
    )
    loaded = SeekDBCollectionAdapter.from_config(config)
    schema = CollectionSchemas.context_collection(collection_name, 4)
    assert loaded.create_collection(
        collection_name,
        schema,
        distance="cosine",
        sparse_weight=0.4,
        index_name="default",
    )
    try:
        yield loaded
    finally:
        try:
            loaded.drop_collection()
        finally:
            loaded.close()


def _record(
    record_id: str,
    *,
    uri: str,
    name: str,
    content: str,
    vector: list[float],
    sparse: dict[str, float],
    account_id: str,
    updated_at: int,
    tags: list[str],
):
    return {
        "id": record_id,
        "uri": uri,
        "type": "file",
        "context_type": "resource",
        "vector": vector,
        "sparse_vector": sparse,
        "created_at": updated_at - 1000,
        "updated_at": updated_at,
        "active_count": 1,
        "level": 2,
        "name": name,
        "description": name,
        "tags": "demo",
        "search_tags": tags,
        "abstract": content[:40],
        "content": content,
        "md5": f"md5-{record_id}",
        "account_id": account_id,
        "owner_user_id": "user-a",
        "acl_mode": "none",
        "acl_direct_grants": [],
        "acl_inherited_grants": [],
    }


def test_full_collection_contract(adapter):
    base = 1_767_225_600_000
    records = [
        _record(
            "id-a",
            uri="viking://resources/docs/a.md",
            name="alpha",
            content="machine learning for Python agents",
            vector=[1, 0, 0, 0],
            sparse={"token-a": 1.0, "1": 0.5},
            account_id="acct-a",
            updated_at=base + 1,
            tags=["python", "agent"],
        ),
        _record(
            "id-b",
            uri="viking://resources/docs/deep/b.md",
            name="beta",
            content="vector database retrieval",
            vector=[0.8, 0.2, 0, 0],
            sparse={"token-b": 1.0},
            account_id="acct-a",
            updated_at=base + 2,
            tags=["database"],
        ),
        _record(
            "id-c",
            uri="viking://resources/other/c.md",
            name="gamma",
            content="unrelated cooking notes",
            vector=[0, 1, 0, 0],
            sparse={"token-c": 1.0},
            account_id="acct-b",
            updated_at=base + 3,
            tags=["cooking"],
        ),
    ]
    assert adapter.upsert(records) == ["id-a", "id-b", "id-c"]

    fetched = adapter.get(["id-c", "missing", "id-a"])
    assert [item["id"] for item in fetched] == ["id-c", "id-a"]

    vector_hits = adapter.query(
        query_vector=[1, 0, 0, 0],
        filter=Eq("account_id", "acct-a"),
        limit=3,
        output_fields=["id", "uri", "account_id"],
    )
    assert [item["id"] for item in vector_hits] == ["id-a", "id-b"]
    assert vector_hits[0]["uri"] == "viking://resources/docs/a.md"

    scoped = adapter.query(
        query_vector=[1, 0, 0, 0],
        filter=And(
            [
                Eq("account_id", "acct-a"),
                PathScope("uri", "viking://resources/docs", depth=1),
            ]
        ),
        limit=10,
    )
    assert [item["id"] for item in scoped] == ["id-a"]

    tag_hits = adapter.query(
        query_vector=[1, 0, 0, 0],
        filter=Contains("search_tags", "python"),
        limit=10,
    )
    assert [item["id"] for item in tag_hits] == ["id-a"]

    range_hits = adapter.query(
        query_vector=[1, 0, 0, 0],
        filter=And(
            [
                Eq("account_id", "acct-a"),
                Range("updated_at", gte=base + 2, lte=base + 3),
            ]
        ),
        limit=10,
    )
    assert [item["id"] for item in range_hits] == ["id-b"]

    nested_filter_hits = adapter.query(
        query_vector=[1, 0, 0, 0],
        filter=In("search_tags", ["database", "cooking"]),
        limit=10,
    )
    assert {item["id"] for item in nested_filter_hits} == {"id-b", "id-c"}

    excluded = adapter.query(
        query_vector=[1, 0, 0, 0],
        filter=RawDSL(
            {
                "op": "must_not",
                "field": "uri",
                "conds": ["/resources/docs/deep"],
                "para": "-d=-1",
            }
        ),
        limit=10,
    )
    assert {item["id"] for item in excluded} == {"id-a", "id-c"}

    sparse_hits = adapter.query(
        sparse_query_vector={"token-a": 1.0},
        limit=3,
    )
    assert sparse_hits
    assert sparse_hits[0]["id"] == "id-a"

    hybrid_hits = adapter.query(
        query_vector=[1, 0, 0, 0],
        sparse_query_vector={"token-a": 1.0},
        limit=3,
    )
    assert hybrid_hits[0]["id"] == "id-a"

    keyword_hits = adapter.search_by_keywords(
        query="machine learning",
        fields=["content"],
        limit=3,
        output_fields=["id", "content"],
    )
    assert keyword_hits
    assert keyword_hits[0]["id"] == "id-a"

    assert adapter.count(Eq("account_id", "acct-a")) == 2

    scalar = adapter.query(
        filter=Eq("account_id", "acct-a"),
        order_by="updated_at",
        order_desc=False,
        limit=10,
        output_fields=["id", "updated_at", "name"],
    )
    assert [item["id"] for item in scalar] == ["id-a", "id-b"]

    decayed = adapter.query(
        query_vector=[1, 0, 0, 0],
        filter=Eq("account_id", "acct-a"),
        limit=2,
        advance={
            "time_decay": {
                "protection": "0",
                "origin": "2026-03-01T00:00:00Z",
            }
        },
    )
    assert len(decayed) == 2
    assert decayed[0]["_time_score"] < 1.0
    assert "updated_at" in decayed[0]

    update = adapter.update_data([{"id": "id-b", "name": "beta-updated"}])
    assert update == ["id-b"]
    updated = adapter.get(["id-b"])[0]
    assert updated["name"] == "beta-updated"
    assert updated["content"] == "vector database retrieval"
    assert updated["vector"] == pytest.approx([0.8, 0.2, 0, 0])
    assert updated["sparse_vector"] == {"token-b": 1.0}

    meta = adapter.get_collection().get_meta_data()
    assert meta["Fields"]
    assert any(item.get("Field") == "content" for item in meta.get("FullText", []))

    reopened = SeekDBCollectionAdapter(
        collection_name=adapter.collection_name,
        seekdb_config=adapter.config,
        index_name=adapter.index_name,
        distance_metric=adapter._distance_metric,
        dimension=4,
        sparse_weight=0.4,
    )
    try:
        assert reopened.collection_exists()
        reopened_hits = reopened.query(
            query_vector=[1, 0, 0, 0],
            filter=Eq("account_id", "acct-a"),
            limit=2,
        )
        assert [item["id"] for item in reopened_hits] == ["id-a", "id-b"]
    finally:
        reopened.close()

    assert adapter.delete(filter=Eq("account_id", "acct-b")) == 1
    assert adapter.count(Eq("account_id", "acct-b")) == 0
    assert adapter.count() == 2

    adapter.upsert(
        [
            {
                "id": "id-b",
                "vector": [0.8, 0.2, 0, 0],
                "content": "",
            }
        ]
    )
    replaced = adapter.get(["id-b"])[0]
    assert "account_id" not in replaced
    assert "name" not in replaced

    adapter.clear()
    assert adapter.count() == 0


def test_version_is_seekdb_server(adapter):
    config = adapter.config
    conn = pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.connection_user(),
        password=config.password,
        database=config.database,
        autocommit=True,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()[0]
        assert "SeekDB" in version or "seekdb" in version
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("distance_metric", "expected_top_id"),
    [
        ("l2", "metric-near"),
        ("ip", "metric-loud"),
    ],
)
def test_dense_distance_metrics(distance_metric, expected_top_id):
    host = os.getenv("SEEKDB_TEST_HOST", "127.0.0.1")
    port = int(os.getenv("SEEKDB_TEST_PORT", "2881"))
    database = os.environ["SEEKDB_TEST_DATABASE"]
    user = os.getenv("SEEKDB_TEST_USER", "root")
    password = os.getenv("SEEKDB_TEST_PASSWORD", "")
    collection_name = f"ov_metric_{distance_metric}_{uuid.uuid4().hex[:8]}"

    config = VectorDBBackendConfig(
        backend="openviking_seekdb.adapter.SeekDBCollectionAdapter",
        name=collection_name,
        index_name="default",
        dimension=4,
        distance_metric=distance_metric,
        sparse_weight=0,
        custom_params={
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
            "pool_size": 2,
            "fulltext": True,
            "fulltext_analyzer": "ik",
        },
    )
    loaded = SeekDBCollectionAdapter.from_config(config)
    schema = CollectionSchemas.context_collection(collection_name, 4)
    assert loaded.create_collection(
        collection_name,
        schema,
        distance=distance_metric,
        sparse_weight=0,
        index_name="default",
    )
    try:
        base = 1_767_225_600_000
        loaded.upsert(
            [
                _record(
                    "metric-near",
                    uri="viking://resources/metric/near.md",
                    name="near",
                    content="near euclidean vector",
                    vector=[1, 0, 0, 0],
                    sparse={},
                    account_id="acct-metric",
                    updated_at=base,
                    tags=[],
                ),
                _record(
                    "metric-loud",
                    uri="viking://resources/metric/loud.md",
                    name="loud",
                    content="large inner product vector",
                    vector=[0, 10, 0, 0],
                    sparse={},
                    account_id="acct-metric",
                    updated_at=base + 1,
                    tags=[],
                ),
            ]
        )
        hits = loaded.query(
            query_vector=[1, 2, 0, 0],
            limit=2,
            output_fields=["id", "name"],
        )
        other_id = "metric-loud" if expected_top_id == "metric-near" else "metric-near"
        assert [item["id"] for item in hits] == [expected_top_id, other_id]
    finally:
        try:
            loaded.drop_collection()
        finally:
            loaded.close()


@pytest.mark.parametrize(
    "vector_index_type",
    ["flat", "hnsw", "hnsw_sq", "hnsw_bq", "ivf_flat", "ivf_pq"],
)
def test_configured_vector_index_types(vector_index_type):
    host = os.getenv("SEEKDB_TEST_HOST", "127.0.0.1")
    port = int(os.getenv("SEEKDB_TEST_PORT", "2881"))
    database = os.environ["SEEKDB_TEST_DATABASE"]
    user = os.getenv("SEEKDB_TEST_USER", "root")
    password = os.getenv("SEEKDB_TEST_PASSWORD", "")
    collection_name = f"ov_idx_{vector_index_type}_{uuid.uuid4().hex[:8]}"
    custom_params = {
        "host": host,
        "port": port,
        "database": database,
        "user": user,
        "password": password,
        "pool_size": 2,
        "fulltext": True,
        "fulltext_analyzer": "ik",
        "vector_index_type": vector_index_type,
        "hnsw_m": 8,
        "hnsw_ef_construction": 32,
        "hnsw_ef_search": 16,
        "ivf_nlist": 2,
        "ivf_sample_per_nlist": 2,
        "ivf_pq_m": 4,
        "ivf_nprobes": 2,
    }
    if vector_index_type == "hnsw_bq":
        custom_params.update(
            {
                "hnsw_refine_k": 2.0,
                "hnsw_refine_type": "sq8",
                "hnsw_bq_bits_query": 32,
                "hnsw_bq_use_fht": True,
            }
        )
    config = VectorDBBackendConfig(
        backend="openviking_seekdb.adapter.SeekDBCollectionAdapter",
        name=collection_name,
        index_name="default",
        dimension=4,
        distance_metric="cosine",
        sparse_weight=0,
        custom_params=custom_params,
    )
    loaded = SeekDBCollectionAdapter.from_config(config)
    schema = CollectionSchemas.context_collection(collection_name, 4)
    assert loaded.create_collection(
        collection_name,
        schema,
        distance="cosine",
        sparse_weight=0,
        index_name="default",
    )
    try:
        base = 1_767_225_600_000
        loaded.upsert(
            [
                _record(
                    "index-a",
                    uri="viking://resources/index/a.md",
                    name="a",
                    content="first index record",
                    vector=[1, 0, 0, 0],
                    sparse={},
                    account_id="acct-index",
                    updated_at=base,
                    tags=[],
                ),
                _record(
                    "index-b",
                    uri="viking://resources/index/b.md",
                    name="b",
                    content="second index record",
                    vector=[0.7, 0.3, 0, 0],
                    sparse={},
                    account_id="acct-index",
                    updated_at=base + 1,
                    tags=[],
                ),
                _record(
                    "index-c",
                    uri="viking://resources/index/c.md",
                    name="c",
                    content="third index record",
                    vector=[0, 1, 0, 0],
                    sparse={},
                    account_id="acct-index",
                    updated_at=base + 2,
                    tags=[],
                ),
            ]
        )
        hits = loaded.query(
            query_vector=[1, 0, 0, 0],
            limit=3,
            output_fields=["id", "name"],
        )
        assert [item["id"] for item in hits] == ["index-a", "index-b", "index-c"]
        index_meta = loaded.get_collection().get_index_meta_data("default")
        assert index_meta["VectorIndex"]["IndexType"] == vector_index_type
        collection_record = loaded._client.load_collection(collection_name)
        assert collection_record is not None
        physical_indexes = loaded._client.table_indexes(collection_record["physical_name"])
        vector_index = derive_index_name(collection_name, "default", "vector")
        assert (vector_index in physical_indexes) is (vector_index_type != "flat")
    finally:
        try:
            loaded.drop_collection()
        finally:
            loaded.close()
