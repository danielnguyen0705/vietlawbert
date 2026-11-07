"""
neo4j_client.py - Tầng điều phối kết nối Neo4j Bolt Driver trung tâm cho VietLawBERT.
Bảo toàn giao dịch ACID, khởi tạo đầy đủ Schema Constraints (Article, Chunk, LawDocument) và xử lý Batch có Consume.
"""

from __future__ import annotations

import time
import logging
from configs.logging_config import get_subsystem_logger
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase, Driver
from configs.config import config

logger = get_subsystem_logger("database", "neo4j_client")


class Neo4jClient:
    """Quản lý kết nối và thực thi truy vấn Cypher phân tán có khả năng chịu lỗi."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        drop_existing: bool = False,
    ):
        self.uri = uri or getattr(config, "NEO4J_URI", "bolt://localhost:7687")
        self.user = user or getattr(config, "NEO4J_USER", "neo4j")
        self.password = password or getattr(config, "NEO4J_PASSWORD", "vietlawbert2026")

        self.driver: Driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60.0,
        )
        self.driver.verify_connectivity()
        logger.info("Kết nối Neo4j Bolt thành công tại %s.", self.uri)
        self.last_dropped = 0
        self._dropped_total = 0

        if drop_existing:
            self.drop_database()
        self.create_constraints()

    def close(self) -> None:
        self.driver.close()
        logger.info("Đã đóng kết nối Neo4j Driver.")

    def drop_database(self, batch_size: int = 10000) -> None:
        # Xóa THEO LÔ (mỗi lô là 1 transaction tự-commit) để tránh OutOfMemory khi graph lớn.
        logger.warning("Đang dọn sạch toàn bộ Neo4j theo lô (batch=%d)...", batch_size)
        total = 0
        with self.driver.session() as session:
            while True:
                c = session.run(
                    "MATCH (n) WITH n LIMIT $b DETACH DELETE n RETURN count(n) AS c",
                    b=batch_size,
                ).single()["c"]
                total += c
                if c == 0:
                    break
        logger.info("Đã dọn sạch Neo4j: xóa %d node.", total)

    def create_constraints(self) -> None:
        """Tạo ràng buộc duy nhất và chỉ mục trên LawDocument, Article, Chunk và Relationship."""
        constraints = [
            "CREATE CONSTRAINT law_doc_id IF NOT EXISTS FOR (d:LawDocument) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.article_id IS UNIQUE",
            "CREATE INDEX law_doc_number IF NOT EXISTS FOR (d:LawDocument) ON (d.doc_number)",
            "CREATE INDEX chunk_macro IF NOT EXISTS FOR (c:Chunk) ON (c.macro_label)",
            "CREATE INDEX chunk_doc_id IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
            "CREATE INDEX legal_rel_type IF NOT EXISTS FOR ()-[r:LEGAL_RELATION]-() ON (r.type)",
        ]
        with self.driver.session() as session:
            for q in constraints:
                try:
                    session.run(q).consume()
                except Exception as exc:
                    logger.debug("Thông báo khởi tạo schema Neo4j: %s", exc)
        logger.info("Đã xác thực toàn bộ Constraints & Indices trên Neo4j.")

    def execute_query(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Thực thi truy vấn Cypher và thu hồi dữ liệu ngay lập tức (khắc phục AttributeError)."""
        with self.driver.session() as session:
            result = session.run(cypher, parameters or {})
            data = result.data()
            result.consume()
            return data

    def execute_write(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        with self.driver.session() as session:
            return session.execute_write(lambda tx: tx.run(cypher, parameters or {}).data())

    def execute_read(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            return session.execute_read(lambda tx: tx.run(cypher, parameters or {}).data())

    def _reconnect(self) -> None:
        """Tái tạo driver khi kết nối Bolt chết giữa chừng (defunct/ServiceUnavailable/timeout)."""
        try:
            self.driver.close()
        except Exception:
            pass
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60.0,
        )
        self.driver.verify_connectivity()
        logger.info("Đã TÁI KẾT NỐI Neo4j driver sau lỗi kết nối.")

    def execute_batch(
        self,
        cypher: str,
        batch: List[Dict[str, Any]],
        batch_size: int = 500,
        max_retries: int = 5,
    ) -> int:
        """
        Ghi theo lô CHỊU LỖI, chống mất dữ liệu âm thầm:
        - MỖI sub-batch mở SESSION RIÊNG: session chết KHÔNG đầu độc các lô sau.
        - Lỗi KẾT NỐI (defunct/ServiceUnavailable/timeout) => TÁI TẠO driver rồi thử lại
          với backoff tăng dần; lỗi DỮ LIỆU thì chỉ thử lại (không tái kết nối).
        - ĐẾM mọi bản ghi bị bỏ (self.last_dropped + tích luỹ self._dropped_total) và
          ghi log ERROR để lần chạy A100 không mất bản ghi mà không ai hay.
        """
        if not batch:
            self.last_dropped = 0
            return 0

        conn_markers = ("defunct", "ServiceUnavailable", "SessionExpired",
                        "connection", "Connection", "timed out", "handshake", "Couldn't connect")
        total, dropped = 0, 0
        for i in range(0, len(batch), batch_size):
            sub_batch = batch[i : i + batch_size]
            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    with self.driver.session() as session:
                        session.run(cypher, parameters={"batch": sub_batch}).consume()
                    total += len(sub_batch)
                    success = True
                    break
                except Exception as exc:
                    msg = str(exc)
                    is_conn = any(m in msg for m in conn_markers)
                    logger.warning(
                        "Lỗi nạp batch Neo4j (Lần %d/%d, %s): %s. Tạm dừng %ds...",
                        attempt, max_retries,
                        "KẾT NỐI" if is_conn else "DỮ LIỆU", msg[:160], attempt * 2,
                    )
                    time.sleep(attempt * 2)
                    if is_conn:
                        try:
                            self._reconnect()
                        except Exception as rexc:
                            logger.warning("Tái kết nối thất bại: %s", str(rexc)[:120])
            if not success:
                dropped += len(sub_batch)
                logger.error("BỎ sub-batch Neo4j (%d bản ghi) sau %d lần thử.",
                             len(sub_batch), max_retries)

        self.last_dropped = dropped
        self._dropped_total = getattr(self, "_dropped_total", 0) + dropped
        if dropped:
            logger.error("⚠ Tổng bản ghi Neo4j bị bỏ (tích luỹ phiên): %d", self._dropped_total)
        return total


Neo4jManager = Neo4jClient
Neo4jStore = Neo4jClient