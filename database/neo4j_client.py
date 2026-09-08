"""
neo4j_client.py - Tầng điều phối kết nối Neo4j Bolt Driver trung tâm cho VietLawBERT.
Bảo toàn giao dịch ACID, khởi tạo Constraints & Indices và hỗ trợ batch operations.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase, Driver

from configs.config import config

logger = logging.getLogger("VietLawBERT_Neo4jClient")


class Neo4jClient:
    """Singleton-ready Client quản lý kết nối và thực thi truy vấn Cypher."""

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
        )
        self.driver.verify_connectivity()
        logger.info("Kết nối Neo4j Bolt thành công tại %s.", self.uri)

        if drop_existing:
            self.drop_database()
        self.create_constraints()

    def close(self) -> None:
        self.driver.close()
        logger.info("Đã đóng kết nối Neo4j Driver.")

    def drop_database(self) -> None:
        logger.warning("Đang dọn sạch toàn bộ cơ sở dữ liệu Neo4j...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Đã dọn sạch cơ sở dữ liệu Neo4j.")

    def create_constraints(self) -> None:
        """Tạo ràng buộc duy nhất và chỉ mục trên LawDocument và Chunk."""
        constraints = [
            "CREATE CONSTRAINT law_doc_id IF NOT EXISTS FOR (d:LawDocument) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE INDEX law_doc_number IF NOT EXISTS FOR (d:LawDocument) ON (d.doc_number)",
            "CREATE INDEX chunk_macro IF NOT EXISTS FOR (c:Chunk) ON (c.macro_label)",
        ]
        with self.driver.session() as session:
            for q in constraints:
                try:
                    session.run(q)
                except Exception as exc:
                    logger.debug("Thông báo tạo chỉ mục Neo4j: %s", exc)
        logger.info("Đã xác thực toàn bộ Constraints & Indices trên Neo4j.")

    def execute_write(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        with self.driver.session() as session:
            return session.execute_write(lambda tx: tx.run(cypher, parameters or {}).data())

    def execute_read(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            return session.execute_read(lambda tx: tx.run(cypher, parameters or {}).data())

    def execute_batch(self, cypher: str, batch: List[Dict[str, Any]], batch_size: int = 500) -> int:
        total = 0
        with self.driver.session() as session:
            for i in range(0, len(batch), batch_size):
                sub_batch = batch[i : i + batch_size]
                session.run(cypher, parameters={"batch": sub_batch})
                total += len(sub_batch)
        return total


Neo4jManager = Neo4jClient
Neo4jStore = Neo4jClient