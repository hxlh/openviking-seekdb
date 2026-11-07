"""Conversion between OpenViking sparse dictionaries and SeekDB SPARSEVECTOR."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SparseVectorCodec:
    """Map arbitrary learned-sparse terms into SeekDB's positive integer dimensions.

    SeekDB sparse vectors are integer-keyed. Learned sparse terms such as ``tok_17``
    are therefore hashed deterministically into ``1..max_dimension``. Collisions
    within one vector are summed, which keeps the dot product deterministic.
    Callers that need to inspect the original dictionary retain it in the nullable
    ``sparse_terms`` JSON column.
    """

    max_dimension: int = 500_000
    max_nonzero: int = 4_096

    def encode(self, value: Any) -> str | None:
        normalized = self.normalize(value)
        if not normalized:
            return None
        if len(normalized) > self.max_nonzero:
            raise ValueError(f"sparse_vector exceeds {self.max_nonzero} non-zero dimensions")
        parts = [
            f"{dimension}:{self._format_float(weight)}" for dimension, weight in normalized.items()
        ]
        return "{" + ",".join(parts) + "}"

    def normalize(self, value: Any) -> dict[int, float]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("sparse_vector must be an object of term -> weight")
        combined: dict[int, float] = {}
        for raw_term, raw_weight in value.items():
            term = str(raw_term)
            if not term:
                raise ValueError("sparse vector terms cannot be empty")
            if len(term.encode("utf-8")) > 256:
                raise ValueError("sparse vector terms cannot exceed 256 UTF-8 bytes")
            if isinstance(raw_weight, bool):
                raise ValueError("sparse vector weights must be finite numbers")
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError("sparse vector weights must be finite numbers") from exc
            if not math.isfinite(weight):
                raise ValueError("sparse vector weights must be finite numbers")
            if weight < 0:
                raise ValueError("SeekDB sparse vector weights cannot be negative")
            if weight == 0:
                continue
            dimension = self._dimension_for_term(term)
            combined[dimension] = combined.get(dimension, 0.0) + weight
        return {dimension: combined[dimension] for dimension in sorted(combined)}

    def _dimension_for_term(self, term: str) -> int:
        stripped = term.strip()
        if stripped.isdigit():
            number = int(stripped)
            if 1 <= number <= self.max_dimension:
                return number
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.max_dimension + 1

    @staticmethod
    def _format_float(value: float) -> str:
        return format(value, ".17g")
