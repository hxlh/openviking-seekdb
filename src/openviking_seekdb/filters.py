"""Compile OpenViking's vector-filter DSL into parameterized SeekDB SQL."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .schema import SchemaSpec, datetime_to_epoch_ms

_DEPTH_RE = re.compile(r"^\s*-d=(-?\d+)\s*$")


@dataclass(frozen=True)
class SQLFilter:
    sql: str
    params: tuple[Any, ...]


class SeekDBFilterCompiler:
    """Translate the exact filter payload emitted by OpenViking v0.5.3."""

    def __init__(self, schema: SchemaSpec, *, alias: str = "r") -> None:
        self.schema = schema
        self.alias = alias

    def compile(self, payload: Optional[Mapping[str, Any]]) -> SQLFilter:
        if not payload:
            return SQLFilter("TRUE", ())
        if not isinstance(payload, Mapping):
            raise ValueError("SeekDB filters must be an object")
        if set(payload) == {"filter"}:
            nested = payload.get("filter")
            if nested is None:
                return SQLFilter("TRUE", ())
            return self.compile(nested)

        op = str(payload.get("op", "")).lower()
        if op in {"and", "or"}:
            return self._compile_logical(op, payload.get("conds", []))
        if op in {"must", "must_not"}:
            return self._compile_must(op, payload)
        if op == "prefix":
            field = self._field(payload)
            expr = self._value_expr(field, string=True)
            prefix = str(payload.get("prefix", ""))
            return SQLFilter(
                f"{expr} LIKE %s ESCAPE '='",
                (self._escape_like(prefix) + "%",),
            )
        if op == "contains":
            field = self._field(payload)
            expr = self._value_expr(field, string=True)
            substring = str(payload.get("substring", ""))
            return SQLFilter(f"LOCATE(%s, {expr}) > 0", (substring,))
        if op in {"range", "time_range"}:
            return self._compile_range(op, payload)
        if op == "range_out":
            inner = self._compile_range("range", payload)
            return SQLFilter(f"NOT ({inner.sql})", inner.params)
        raise NotImplementedError(f"SeekDB filter op={op!r} is not supported")

    def _compile_logical(self, op: str, conditions: Any) -> SQLFilter:
        if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
            raise ValueError(f"SeekDB filter {op}.conds must be a list")
        fragments: list[str] = []
        params: list[Any] = []
        for condition in conditions:
            if not isinstance(condition, Mapping):
                raise ValueError(f"SeekDB filter {op} branches must be objects")
            compiled = self.compile(condition)
            if compiled.sql == "TRUE":
                continue
            if compiled.sql == "FALSE":
                if op == "and":
                    return SQLFilter("FALSE", ())
                continue
            fragments.append(f"({compiled.sql})")
            params.extend(compiled.params)
        if not fragments:
            # A vacuous AND is true; a vacuous OR is a contradiction.
            return SQLFilter("TRUE" if op == "and" else "FALSE", ())
        return SQLFilter(f" {op.upper()} ".join(fragments), tuple(params))

    def _compile_must(self, op: str, payload: Mapping[str, Any]) -> SQLFilter:
        field = self._field(payload)
        field_spec = self.schema.fields[field]
        conditions = payload.get("conds", [])
        if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
            raise ValueError(f"SeekDB {op}.conds must be a list")
        conditions = list(conditions)
        if not conditions:
            return SQLFilter("FALSE" if op == "must" else "TRUE", ())

        depth = self._parse_depth(
            payload.get("para", "-d=0") if field_spec.field_type == "path" else payload.get("para")
        )
        if field_spec.field_type == "path":
            path_clauses: list[str] = []
            params: list[Any] = []
            for condition in conditions:
                clause, clause_params = self._path_scope(field, str(condition), depth)
                path_clauses.append(f"({clause})")
                params.extend(clause_params)
            clause = " OR ".join(path_clauses)
            if op == "must_not":
                clause = f"NOT ({clause})"
            return SQLFilter(clause, tuple(params))

        if payload.get("para") not in (None, ""):
            raise ValueError("filter para is only supported for path fields")

        if field_spec.field_type == "date_time":
            conditions = [datetime_to_epoch_ms(value) for value in conditions]

        if field in {"id", "content"}:
            expr = self._value_expr(field)
            placeholders = ", ".join(["%s"] * len(conditions))
            operator = "IN" if op == "must" else "NOT IN"
            return SQLFilter(f"{expr} {operator} ({placeholders})", tuple(conditions))

        metadata_expr = self._json_expr(field)
        conditions_json = json.dumps(conditions, ensure_ascii=False, allow_nan=False)
        if op == "must":
            return SQLFilter(
                f"JSON_OVERLAPS({metadata_expr}, CAST(%s AS JSON))",
                (conditions_json,),
            )
        return SQLFilter(
            f"COALESCE(NOT JSON_OVERLAPS({metadata_expr}, CAST(%s AS JSON)), TRUE)",
            (conditions_json,),
        )

    def _compile_range(self, op: str, payload: Mapping[str, Any]) -> SQLFilter:
        field = self._field(payload)
        spec = self.schema.fields[field]
        expr = self._value_expr(field)
        fragments: list[str] = []
        params: list[Any] = []
        for key, sql_op in (("gte", ">="), ("gt", ">"), ("lte", "<="), ("lt", "<")):
            if key not in payload or payload[key] is None:
                continue
            value = payload[key]
            if spec.field_type == "date_time":
                value = datetime_to_epoch_ms(value)
            fragments.append(f"{expr} {sql_op} %s")
            params.append(value)
        if not fragments:
            return SQLFilter("TRUE" if op == "time_range" else "FALSE", ())
        return SQLFilter(" AND ".join(fragments), tuple(params))

    def _path_scope(
        self, field: str, path: str, depth: Optional[int]
    ) -> tuple[str, tuple[Any, ...]]:
        normalized = self._normalize_path(path)
        raw_expr = self._value_expr(field, string=True)
        field_expr = (
            f"(CASE WHEN {raw_expr} IS NULL THEN NULL "
            f"WHEN TRIM({raw_expr}) = '' OR TRIM(TRAILING '/' FROM TRIM({raw_expr})) = '' "
            f"THEN '/' ELSE TRIM(TRAILING '/' FROM TRIM({raw_expr})) END)"
        )
        if depth == 0:
            return f"{field_expr} = %s", (normalized,)

        child_pattern = "/%" if normalized == "/" else self._escape_like(normalized) + "/%"
        if depth is None or depth < 0:
            return (
                f"({field_expr} = %s OR {field_expr} LIKE %s ESCAPE '=')",
                (normalized, child_pattern),
            )
        start_pos = 2 if normalized == "/" else len(normalized) + 2
        suffix = f"SUBSTRING({field_expr}, {start_pos})"
        relative_depth = f"(CHAR_LENGTH({suffix}) - CHAR_LENGTH(REPLACE({suffix}, '/', '')) + 1)"
        return (
            f"({field_expr} = %s OR ({field_expr} LIKE %s ESCAPE '=' AND {relative_depth} <= %s))",
            (normalized, child_pattern, depth),
        )

    def _field(self, payload: Mapping[str, Any]) -> str:
        field = payload.get("field")
        if not isinstance(field, str):
            raise ValueError("SeekDB filter field must be a string")
        if field not in self.schema.fields:
            raise ValueError(f"unknown SeekDB filter field {field!r}")
        if self.schema.fields[field].field_type in {"vector", "sparse_vector"}:
            raise ValueError(f"field {field!r} cannot be used as a scalar filter")
        return field

    def _json_expr(self, field: str) -> str:
        self._validate_path_name(field)
        return f"JSON_EXTRACT({self.alias}.`metadata`, '$.{field}')"

    def _value_expr(self, field: str, *, string: bool = False) -> str:
        if field == "id":
            expr = f"{self.alias}.`id`"
        elif field == self.schema.content_field:
            expr = f"{self.alias}.`content`"
        else:
            expr = self._json_expr(field)
            if string:
                expr = (
                    f"CAST(JSON_UNQUOTE(JSON_EXTRACT({self.alias}.`metadata`, "
                    f"'$.{field}')) AS CHAR)"
                )
        if string and field in {"id", "content"}:
            expr = f"CAST({expr} AS CHAR)"
        return expr

    @staticmethod
    def _parse_depth(value: Any) -> Optional[int]:
        if value in (None, ""):
            return -1
        if not isinstance(value, str):
            raise ValueError("path filter para must look like '-d=<integer>'")
        match = _DEPTH_RE.fullmatch(value)
        if not match:
            raise ValueError("path filter para must look like '-d=<integer>'")
        return int(match.group(1))

    @staticmethod
    def _normalize_path(value: str) -> str:
        stripped = str(value).strip() or "/"
        return stripped.rstrip("/") or "/"

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("=", "==").replace("%", "=%").replace("_", "=_")

    @staticmethod
    def _validate_path_name(field: str) -> None:
        if not field.isascii() or not field.replace("_", "").isalnum() or field[0].isdigit():
            raise ValueError(f"unsafe JSON field name {field!r}")
