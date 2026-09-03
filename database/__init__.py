"""
vietlawbert.database
~~~~~~~~~~~~~~~~~~~~
Hạ tầng lưu trữ lai Hybrid Storage cho hệ thống VietLawBERT:
- Vector Database: Quản lý Milvus Client, schema mở rộng và chỉ mục HNSW Cosine.
- Graph Database: Quản lý Neo4j Driver, APOC dynamic transactions và ontology pháp lý.
- Cypher Templates: Thư viện truy vấn phục vụ Multi-hop Reasoning và GG-SLM Triplet Mining.
"""

from .milvus_client import MilvusClientWrapper, MilvusStore
from .neo4j_client import Neo4jClient, Neo4jManager, Neo4jStore
from .cypher_templates import (
    TRACE_HIERARCHICAL_ROOT,
    FIND_CASCADE_IMPACT,
    GET_NEIGHBORS,
    GG_SLM_TRIPLET_MINING,
    get_trace_hierarchical_root,
    get_find_cascade_impact,
    get_neighbors,
    get_gg_slm_triplets,
)

__all__ = [
    "MilvusClientWrapper",
    "MilvusStore",
    "Neo4jClient",
    "Neo4jManager",
    "Neo4jStore",
    "TRACE_HIERARCHICAL_ROOT",
    "FIND_CASCADE_IMPACT",
    "GET_NEIGHBORS",
    "GG_SLM_TRIPLET_MINING",
    "get_trace_hierarchical_root",
    "get_find_cascade_impact",
    "get_neighbors",
    "get_gg_slm_triplets",
]