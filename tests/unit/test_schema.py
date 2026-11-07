import pytest

from openviking_seekdb.errors import SeekDBConfigurationError
from openviking_seekdb.schema import (
    SchemaSpec,
    derive_index_name,
    derive_physical_name,
    normalize_field_value,
)


def test_openviking_context_schema_is_normalized(schema_dict):
    schema = SchemaSpec.from_mapping(schema_dict)
    assert schema.primary_key == "id"
    assert schema.vector_field == "vector"
    assert schema.sparse_vector_field == "sparse_vector"
    assert schema.content_field == "content"
    assert schema.dimension == 4
    assert "uri" in schema.path_fields
    assert "search_tags" in schema.array_fields
    assert "updated_at" in schema.date_time_fields


def test_schema_rejects_unknown_scalar_index(schema_dict):
    schema_dict["ScalarIndex"].append("missing")
    with pytest.raises(SeekDBConfigurationError, match="unknown field"):
        SchemaSpec.from_mapping(schema_dict)


def test_physical_names_are_bounded_and_deterministic():
    logical = "a" * 200
    physical = derive_physical_name(logical)
    assert physical == derive_physical_name(logical)
    assert len(physical) <= 64
    assert derive_index_name(logical, "default", "vector").startswith("ov_v_")


def test_date_time_accepts_iso_and_epoch(schema_dict):
    schema = SchemaSpec.from_mapping(schema_dict)
    updated = schema.fields["updated_at"]
    assert normalize_field_value(updated, "2026-01-01T00:00:00Z") == 1767225600000
    assert normalize_field_value(updated, 1234) == 1234
