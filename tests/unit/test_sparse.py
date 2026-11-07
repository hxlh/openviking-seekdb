import pytest

from openviking_seekdb.sparse import SparseVectorCodec


def test_sparse_encoding_is_deterministic_and_numeric_keys_are_preserved():
    codec = SparseVectorCodec(max_dimension=1000)
    first = codec.encode({"1": 0.5, "token-a": 1.25, "token-b": 2})
    second = codec.encode({"token-a": 1.25, "token-b": 2, "1": 0.5})
    assert first == second
    assert first is not None
    assert first.startswith("{1:0.5")
    assert "," in first


def test_sparse_zero_weights_are_removed():
    codec = SparseVectorCodec()
    assert codec.encode({"a": 0, "b": 1}) is not None
    assert codec.normalize({"a": 0}) == {}


def test_sparse_rejects_negative_and_non_finite_weights():
    codec = SparseVectorCodec()
    with pytest.raises(ValueError, match="negative"):
        codec.normalize({"a": -1})
    with pytest.raises(ValueError, match="finite"):
        codec.normalize({"a": float("nan")})


def test_sparse_nonzero_limit_is_enforced():
    codec = SparseVectorCodec(max_nonzero=1)
    with pytest.raises(ValueError, match="non-zero"):
        codec.encode({"a": 1, "b": 2})
