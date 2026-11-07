"""Thread-safe PyMySQL connection pool and SeekDB catalog access."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from contextlib import contextmanager, suppress
from typing import Any, Iterator, Mapping, Optional

import pymysql
from pymysql.cursors import DictCursor

from .config import SeekDBConfig
from .errors import SeekDBConnectionError
from .schema import derive_internal_tables, validate_identifier

logger = logging.getLogger(__name__)


class _ConnectionPool:
    def __init__(self, config: SeekDBConfig) -> None:
        self.config = config
        self._available: queue.LifoQueue[Any] = queue.LifoQueue(maxsize=config.pool_size)
        self._slots = threading.BoundedSemaphore(config.pool_size)
        self._lock = threading.Lock()
        self._created = 0
        self._closed = False

    def acquire(self):
        if self._closed:
            raise SeekDBConnectionError("SeekDB connection pool is closed")
        if not self._slots.acquire(timeout=self.config.connect_timeout):
            raise SeekDBConnectionError("Timed out waiting for a SeekDB connection")
        connection = None
        try:
            try:
                connection = self._available.get_nowait()
            except queue.Empty:
                connection = self._open_with_retry()
                with self._lock:
                    self._created += 1
            try:
                connection.ping(False)
            except Exception:
                self._discard(connection)
                connection = None
                connection = self._open_with_retry()
                with self._lock:
                    self._created += 1
            return connection
        except Exception:
            if connection is not None:
                self._discard(connection)
            self._slots.release()
            raise

    def release(self, connection, *, broken: bool = False) -> None:
        if connection is None:
            return
        if broken or self._closed:
            self._discard(connection)
        else:
            try:
                connection.rollback()
                connection.ping(reconnect=False)
                self._available.put_nowait(connection)
            except Exception:
                self._discard(connection)
        self._slots.release()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        while True:
            try:
                connection = self._available.get_nowait()
            except queue.Empty:
                break
            self._close_quietly(connection)
            with self._lock:
                self._created = max(0, self._created - 1)

    def _discard(self, connection) -> None:
        self._close_quietly(connection)
        with self._lock:
            self._created = max(0, self._created - 1)

    def _open_with_retry(self):
        last_error: Optional[BaseException] = None
        for attempt in range(self.config.connect_retries):
            if self._closed:
                raise SeekDBConnectionError("SeekDB connection pool is closed")
            try:
                return pymysql.connect(
                    host=self.config.host,
                    port=self.config.port,
                    user=self.config.connection_user(),
                    password=self.config.password,
                    database=self.config.database,
                    charset=self.config.charset,
                    cursorclass=DictCursor,
                    autocommit=False,
                    connect_timeout=self.config.connect_timeout,
                    read_timeout=self.config.read_timeout,
                    write_timeout=self.config.write_timeout,
                    ssl=self.config.ssl_options(),
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.config.connect_retries:
                    time.sleep(min(2**attempt * 0.1, 1.0))
        raise SeekDBConnectionError(
            f"Unable to connect to SeekDB {self.config.host}:{self.config.port}/"
            f"{self.config.database}: {last_error}"
        ) from last_error

    @staticmethod
    def _close_quietly(connection) -> None:
        with suppress(Exception):
            connection.close()


class SeekDBClient:
    """Minimal official-SQL client used by the OpenViking adapter."""

    def __init__(self, config: SeekDBConfig) -> None:
        self.config = config
        self._pool = _ConnectionPool(config)
        self._catalog_lock = threading.Lock()
        self._catalog_ready = False
        self._collections_table, self._indexes_table = derive_internal_tables(
            config.metadata_prefix
        )
        self._refresh_supported: Optional[bool] = None

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        connection = self._pool.acquire()
        broken = False
        try:
            yield connection
            connection.commit()
        except Exception:
            broken = _is_connection_error()
            try:
                connection.rollback()
            except Exception:
                broken = True
            raise
        finally:
            self._pool.release(connection, broken=broken)

    def ensure_catalog(self) -> None:
        if self._catalog_ready:
            return
        with self._catalog_lock:
            if self._catalog_ready:
                return
            collections_table = self._quote(self._collections_table)
            indexes_table = self._quote(self._indexes_table)
            with self.transaction() as connection, connection.cursor() as cursor:
                cursor.execute(
                    f"CREATE TABLE IF NOT EXISTS {collections_table} ("
                    "`collection_name` VARCHAR(128) NOT NULL, "
                    "`physical_name` VARCHAR(64) NOT NULL, "
                    "`schema_json` LONGTEXT NOT NULL, "
                    "`created_at` BIGINT NOT NULL, "
                    "`updated_at` BIGINT NOT NULL, "
                    "PRIMARY KEY (`collection_name`)"
                    ") DEFAULT CHARSET = utf8mb4"
                )
                cursor.execute(
                    f"CREATE TABLE IF NOT EXISTS {indexes_table} ("
                    "`collection_name` VARCHAR(128) NOT NULL, "
                    "`index_name` VARCHAR(128) NOT NULL, "
                    "`meta_json` LONGTEXT NOT NULL, "
                    "`created_at` BIGINT NOT NULL, "
                    "`updated_at` BIGINT NOT NULL, "
                    "PRIMARY KEY (`collection_name`, `index_name`)"
                    ") DEFAULT CHARSET = utf8mb4"
                )
            self._catalog_ready = True

    def load_collection(self, collection_name: str) -> Optional[dict[str, Any]]:
        self.ensure_catalog()
        with self.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT `physical_name`, `schema_json` FROM "
                f"{self._quote(self._collections_table)} WHERE `collection_name` = %s",
                (collection_name,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return {
            "collection_name": collection_name,
            "physical_name": row["physical_name"],
            "schema": _loads(row["schema_json"]),
        }

    def save_collection(
        self, collection_name: str, physical_name: str, schema: Mapping[str, Any]
    ) -> None:
        self.ensure_catalog()
        now = int(time.time() * 1000)
        payload = json.dumps(schema, ensure_ascii=False, allow_nan=False)
        with self.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT 1 FROM {self._quote(self._collections_table)} "
                "WHERE `collection_name` = %s FOR UPDATE",
                (collection_name,),
            )
            if cursor.fetchone():
                cursor.execute(
                    f"UPDATE {self._quote(self._collections_table)} SET "
                    "`physical_name` = %s, `schema_json` = %s, `updated_at` = %s "
                    "WHERE `collection_name` = %s",
                    (physical_name, payload, now, collection_name),
                )
            else:
                cursor.execute(
                    f"INSERT INTO {self._quote(self._collections_table)} "
                    "(`collection_name`, `physical_name`, `schema_json`, `created_at`, `updated_at`) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (collection_name, physical_name, payload, now, now),
                )

    def delete_collection_record(self, collection_name: str) -> None:
        self.ensure_catalog()
        with self.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {self._quote(self._indexes_table)} WHERE `collection_name` = %s",
                (collection_name,),
            )
            cursor.execute(
                f"DELETE FROM {self._quote(self._collections_table)} WHERE `collection_name` = %s",
                (collection_name,),
            )

    def load_index(self, collection_name: str, index_name: str) -> Optional[dict[str, Any]]:
        self.ensure_catalog()
        with self.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT `meta_json` FROM {self._quote(self._indexes_table)} "
                "WHERE `collection_name` = %s AND `index_name` = %s",
                (collection_name, index_name),
            )
            row = cursor.fetchone()
        return _loads(row["meta_json"]) if row else None

    def list_indexes(self, collection_name: str) -> list[tuple[str, dict[str, Any]]]:
        self.ensure_catalog()
        with self.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT `index_name`, `meta_json` FROM {self._quote(self._indexes_table)} "
                "WHERE `collection_name` = %s ORDER BY `index_name`",
                (collection_name,),
            )
            rows = cursor.fetchall()
        return [(row["index_name"], _loads(row["meta_json"])) for row in rows]

    def save_index(self, collection_name: str, index_name: str, meta: Mapping[str, Any]) -> None:
        self.ensure_catalog()
        now = int(time.time() * 1000)
        payload = json.dumps(meta, ensure_ascii=False, allow_nan=False)
        with self.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT 1 FROM {self._quote(self._indexes_table)} "
                "WHERE `collection_name` = %s AND `index_name` = %s FOR UPDATE",
                (collection_name, index_name),
            )
            if cursor.fetchone():
                cursor.execute(
                    f"UPDATE {self._quote(self._indexes_table)} SET `meta_json` = %s, "
                    "`updated_at` = %s WHERE `collection_name` = %s AND `index_name` = %s",
                    (payload, now, collection_name, index_name),
                )
            else:
                cursor.execute(
                    f"INSERT INTO {self._quote(self._indexes_table)} "
                    "(`collection_name`, `index_name`, `meta_json`, `created_at`, `updated_at`) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (collection_name, index_name, payload, now, now),
                )

    def delete_index(self, collection_name: str, index_name: str) -> None:
        self.ensure_catalog()
        with self.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {self._quote(self._indexes_table)} "
                "WHERE `collection_name` = %s AND `index_name` = %s",
                (collection_name, index_name),
            )

    def table_exists(self, physical_name: str, cursor=None) -> bool:
        validate_identifier(physical_name, kind="physical table name")
        if cursor is not None:
            cursor.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = DATABASE() AND table_name = %s",
                (physical_name,),
            )
            return cursor.fetchone() is not None
        with self.transaction() as connection, connection.cursor() as own_cursor:
            return self.table_exists(physical_name, cursor=own_cursor)

    def table_indexes(self, physical_name: str) -> dict[str, str]:
        validate_identifier(physical_name, kind="physical table name")
        with self.transaction() as connection, connection.cursor() as cursor:
            cursor.execute(f"SHOW INDEX FROM {self._quote(physical_name)}")
            rows = cursor.fetchall()
        result: dict[str, str] = {}
        for row in rows:
            name = str(row.get("Key_name") or row.get("key_name") or "")
            if not name:
                continue
            index_type = str(
                row.get("Index_type") or row.get("index_type") or row.get("Comment") or ""
            ).upper()
            result[name] = index_type
        return result

    def refresh_indexes(self) -> None:
        """Flush SeekDB >=1.3 asynchronous vector/full-text index updates."""
        if not self.config.refresh_after_write:
            return
        with self.transaction() as connection, connection.cursor() as cursor:
            if self._refresh_supported is None:
                cursor.execute("SELECT VERSION() AS version")
                row = cursor.fetchone() or {}
                version = str(row.get("version") or "")
                self._refresh_supported = _supports_index_refresh(version)
            if self._refresh_supported:
                cursor.execute("CALL dbms_index_manager.refresh()")

    def health_check(self) -> bool:
        try:
            with self.transaction() as connection, connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                return cursor.fetchone() is not None
        except Exception as exc:
            logger.warning("SeekDB health check failed: %s", exc)
            return False

    def close(self) -> None:
        self._pool.close()

    @staticmethod
    def _quote(identifier: str) -> str:
        validate_identifier(identifier)
        return f"`{identifier}`"


def _loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8")
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("SeekDB catalog metadata must contain a JSON object")
    return parsed


def _supports_index_refresh(version: str) -> bool:
    import re

    match = re.search(r"seekdb-v?(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?", version, re.I)
    if not match:
        return False
    parts = tuple(int(part or 0) for part in match.groups())
    return parts >= (1, 3, 0, 0)


def _is_connection_error() -> bool:
    # Decided below by inspecting the active exception without swallowing it.
    import sys

    exc = sys.exc_info()[1]
    if exc is None:
        return False
    if isinstance(exc, (pymysql.err.OperationalError, pymysql.err.InterfaceError)):
        code = exc.args[0] if exc.args and isinstance(exc.args[0], int) else None
        return code in {0, 2002, 2003, 2006, 2013, 2055} or code is None
    return False
