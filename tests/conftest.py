"""conftest.py - Cấu hình & fixture dùng chung cho toàn bộ test suite 5 tầng.

- Nối sys.path portable: gốc repo (cho preprocess/rag/training/...), thư mục tests
  (cho _common.*) và harness RQ (cho `import providers.*` của Tier 4/5).
- Khóa CPU + HuggingFace offline để ổn định trên máy trạm (GTX 1050 Ti không hỗ trợ kernel mới).
- Fixture dịch vụ (neo4j/qdrant/es) tự SKIP khi dịch vụ chưa bật, thay vì làm đỏ cả suite.
"""
import os
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
HARNESS_DIR = TESTS_DIR / "rq_experiments_promptfoo"

for _p in (REPO_ROOT, TESTS_DIR, HARNESS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Môi trường suy luận tất định trên máy trạm
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("EMBED_DEVICE", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Nạp .env ở gốc repo để khớp docker-compose (Neo4j/Qdrant/ES) khi chạy pytest trực tiếp trên host
_env = REPO_ROOT / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env, override=False)
    except Exception:
        for _line in _env.read_text(encoding="utf-8", errors="ignore").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from _common import paths as P  # noqa: E402


@pytest.fixture(scope="session")
def project_paths():
    return {
        "root": P.REPO_ROOT,
        "data": P.DATA_ROOT,
        "raw_shards": P.RAW_SHARDS,
        "processed_parquet": P.PROCESSED_PARQUET,
        "bulk_vectors": P.BULK_VECTORS,
        "checkpoints": P.CHECKPOINTS_DIR,
        "datasets": P.DATASETS_DIR,
    }


@pytest.fixture(scope="session")
def neo4j_driver():
    """Driver Neo4j Bolt; SKIP nếu thiếu package hoặc không kết nối được."""
    try:
        from neo4j import GraphDatabase
    except Exception:
        pytest.skip("Chưa cài package neo4j (pip install neo4j).")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD", "vietlawbert2026")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        with driver.session() as s:
            s.run("RETURN 1 AS ping").consume()
    except Exception as e:
        pytest.skip(f"Neo4j không sẵn sàng tại {uri}: {e}")
    yield driver
    driver.close()


@pytest.fixture(scope="session")
def qdrant_client():
    """Client Qdrant; SKIP nếu thiếu package hoặc engine chưa bật."""
    try:
        from qdrant_client import QdrantClient
    except Exception:
        pytest.skip("Chưa cài package qdrant-client.")
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    try:
        client = QdrantClient(host=host, port=port, timeout=60.0)
        client.get_collections()
    except Exception as e:
        pytest.skip(f"Qdrant không sẵn sàng tại {host}:{port}: {e}")
    yield client


@pytest.fixture(scope="session")
def es_client():
    """Client Elasticsearch; SKIP nếu thiếu package hoặc service chưa bật."""
    try:
        from elasticsearch import Elasticsearch
    except Exception:
        pytest.skip("Chưa cài package elasticsearch.")
    url = os.environ.get("ES_HOST") or os.environ.get("ES_URL", "http://localhost:9200")
    try:
        es = Elasticsearch(url, request_timeout=30)
        if not es.ping():
            pytest.skip(f"Elasticsearch không phản hồi ping tại {url}.")
    except Exception as e:
        pytest.skip(f"Elasticsearch không sẵn sàng tại {url}: {e}")
    yield es
