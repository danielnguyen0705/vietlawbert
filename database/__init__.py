"""
vietlawbert.database
~~~~~~~~~~~~~~~~~~~~
Hạ tầng lưu trữ lai Hybrid Storage cho hệ thống VietLawBERT (Kiến trúc v3):
- Vector Database: Quản lý Qdrant Client Wrapper (lát cắt MRL d=256, HNSW Cosine, Payload Filtering).
- Graph Database: Quản lý Neo4j Driver Client, mạng dị thể HIN (22 quan hệ pháp lý và phân cấp AST).
- Cypher Templates: Thư viện truy vấn HIN-Guided Contrastive Triplet Mining và Cascade Impact Analysis.
"""

from .qdrant_client import QdrantClientWrapper
from .neo4j_client import Neo4jClient, Neo4jManager, Neo4jStore
from .build_hin_graph import HINGraphBuilder, run_build_hin
from .cypher_templates import (
    TRACE_HIERARCHICAL_ROOT,
    FIND_CASCADE_IMPACT,
    GET_NEIGHBORS,
    HIN_TRIPLET_MINING,
    get_trace_hierarchical_root,
    get_find_cascade_impact,
    get_neighbors,
    get_hin_triplets,
)

__all__ = [
    # Vector Engine
    "QdrantClientWrapper",
    # Graph Engine
    "Neo4jClient",
    "Neo4jManager",
    "Neo4jStore",
    "HINGraphBuilder",
    "run_build_hin",
    # Cypher Queries
    "TRACE_HIERARCHICAL_ROOT",
    "FIND_CASCADE_IMPACT",
    "GET_NEIGHBORS",
    "HIN_TRIPLET_MINING",
    "get_trace_hierarchical_root",
    "get_find_cascade_impact",
    "get_neighbors",
    "get_hin_triplets",
]