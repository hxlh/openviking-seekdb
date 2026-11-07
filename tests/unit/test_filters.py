from openviking.storage.expr import (
    And,
    Contains,
    Eq,
    In,
    Or,
    PathScope,
    Range,
)
from openviking.storage.vectordb_adapters.base import CollectionAdapter

from openviking_seekdb.filters import SeekDBFilterCompiler
from openviking_seekdb.schema import SchemaSpec


class _CompilerAdapter(CollectionAdapter):
    mode = "seekdb"

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    def _load_existing_collection_if_needed(self):
        raise NotImplementedError

    def _create_backend_collection(self, meta):
        raise NotImplementedError


def _compiler(schema_dict):
    return SeekDBFilterCompiler(SchemaSpec.from_mapping(schema_dict))


def test_path_scope_depth_is_not_equality(schema_dict):
    adapter = _CompilerAdapter("context")
    payload = adapter._compile_filter(PathScope("uri", "viking://resources/docs", depth=1))
    compiled = _compiler(schema_dict).compile(payload)
    assert "LIKE" in compiled.sql
    assert "relative" not in compiled.sql
    assert "SUBSTRING" in compiled.sql
    assert compiled.params == ("/resources/docs", "/resources/docs/%", 1)


def test_empty_or_is_false_and_empty_and_is_true(schema_dict):
    compiler = _compiler(schema_dict)
    assert compiler.compile({"op": "or", "conds": []}).sql == "FALSE"
    assert compiler.compile({"op": "and", "conds": []}).sql == "TRUE"


def test_array_membership_uses_json_overlaps(schema_dict):
    adapter = _CompilerAdapter("context")
    payload = adapter._compile_filter(In("search_tags", ["python", "codex"]))
    compiled = _compiler(schema_dict).compile(payload)
    assert "JSON_OVERLAPS" in compiled.sql
    assert compiled.params == ('["python", "codex"]',)


def test_contains_string_list_and_scalar(schema_dict):
    adapter = _CompilerAdapter("context")
    payload = adapter._compile_filter(Contains("search_tags", "python"))
    compiled = _compiler(schema_dict).compile(payload)
    assert "LOCATE" in compiled.sql
    assert compiled.params == ("python",)


def test_date_range_is_epoch_ms(schema_dict):
    adapter = _CompilerAdapter("context")
    payload = adapter._compile_filter(
        Range("updated_at", gte="2026-01-01T00:00:00Z", lt=1767225600001)
    )
    compiled = _compiler(schema_dict).compile(payload)
    assert compiled.params == (1767225600000, 1767225600001)


def test_nested_and_or(schema_dict):
    adapter = _CompilerAdapter("context")
    expression = And(
        [
            Eq("account_id", "acct-a"),
            Or([Eq("level", 0), Eq("level", 1)]),
        ]
    )
    compiled = _compiler(schema_dict).compile(adapter._compile_filter(expression))
    assert " AND " in compiled.sql
    assert " OR " in compiled.sql
    assert compiled.params == ('["acct-a"]', "[0]", "[1]")
