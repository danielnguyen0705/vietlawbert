"""
retrieval_provider.py - Custom Provider kết nối động cơ LegalRetriever (vietlawbert).
Tương thích hoàn toàn với schema vietlawbench_promptfoo_tests.json:
  - Trả về retrieved_chunks (đo chunk_hit@5)
  - Trả về retrieved_docs (đo doc_hit@5)
  - Trả về top_contexts (phục vụ LLM-as-a-Judge)
Sử dụng chính xác các phương thức nội tại: _search_dense, _search_sparse_es, _rrf_fusion, retrieve.
"""

import json
import os
import sys
import time
from pathlib import Path
import qdrant_client

# Monkeypatch phòng thủ: loại bỏ check_compatibility khỏi mọi lời gọi QdrantClient
_original_qdrant_init = qdrant_client.QdrantClient.__init__
def _safe_qdrant_init(self, *args, **kwargs):
    kwargs.pop("check_compatibility", None)
    return _original_qdrant_init(self, *args, **kwargs)
qdrant_client.QdrantClient.__init__ = _safe_qdrant_init

# Ép chạy CPU an toàn để tránh xung đột GPU GTX 1050 Ti
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["EMBED_DEVICE"] = "cpu"

# Khóa mạng HuggingFace Offline mode để triệt tiêu hoàn toàn các HTTP calls kiểm tra cache gây nghẽn tiến trình
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

VIETLAWBERT_ROOT = Path(os.environ.get("VIETLAWBERT_ROOT") or next((str(p) for p in Path(__file__).resolve().parents if (p / "rag").is_dir()), str(Path(__file__).resolve().parent)))
if str(VIETLAWBERT_ROOT) not in sys.path:
    sys.path.insert(0, str(VIETLAWBERT_ROOT))

_GLOBAL_RETRIEVER = None


def get_retriever():
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        try:
            from rag.retriever import LegalRetriever
            _GLOBAL_RETRIEVER = LegalRetriever()
        except ImportError:
            from rag.retriever import LegalHybridRetriever
            _GLOBAL_RETRIEVER = LegalHybridRetriever()

        # Thực hiện Warm-up truy vấn để thiết lập sẵn Connection Pool và nạp cache Lucene/Qdrant
        try:
            if hasattr(_GLOBAL_RETRIEVER, "_search_sparse_es"):
                _GLOBAL_RETRIEVER._search_sparse_es("khởi động hệ thống", top_k=1)
            if hasattr(_GLOBAL_RETRIEVER, "_search_dense"):
                _GLOBAL_RETRIEVER._search_dense("khởi động hệ thống", top_k=1)
        except Exception:
            pass

    return _GLOBAL_RETRIEVER


def _extract_doc_info(item: dict):
    """Trích xuất an toàn chunk_id, số hiệu văn bản và nội dung từ kết quả truy xuất."""
    if not isinstance(item, dict):
        return "", "", ""
    chunk_id = str(item.get("chunk_id") or item.get("id") or "").strip()
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    doc_number = str(
        item.get("doc_number")
        or item.get("so_hieu")
        or metadata.get("doc_number")
        or metadata.get("so_hieu")
        or item.get("doc_id")
        or ""
    ).strip()
    text = str(item.get("text") or item.get("content") or metadata.get("content") or "").strip()
    return chunk_id, doc_number, text


def call_api(prompt: str, options: dict = None, context: dict = None) -> dict:
    """Entrypoint cho Promptfoo và Pytest."""
    options = options or {}
    config = options.get("config", {})
    mode = config.get("mode", "full_architecture")
    top_k = config.get("top_k", 5)

    retriever = get_retriever()
    query = prompt.strip()

    start_time = time.perf_counter()

    try:
        # 1. Nhánh Dense-Only: Sử dụng phương thức _search_dense chuẩn xác
        if mode == "dense_only":
            if hasattr(retriever, "_search_dense"):
                results = retriever._search_dense(query, top_k=top_k)
            else:
                results = retriever._search_dense_qdrant(query, top_k=top_k)

        # 2. Nhánh Sparse-Only BM25: Gọi _search_sparse_es
        elif mode == "sparse_only":
            results = retriever._search_sparse_es(query, top_k=top_k)

        # 3. Nhánh Hybrid RRF: Dung hợp Dense và Sparse qua _rrf_fusion
        elif mode == "hybrid_rrf":
            if hasattr(retriever, "_search_dense"):
                dense_res = retriever._search_dense(query, top_k=top_k * 2)
            else:
                dense_res = retriever._search_dense_qdrant(query, top_k=top_k * 2)

            sparse_res = retriever._search_sparse_es(query, top_k=top_k * 2)

            if hasattr(retriever, "_rrf_fusion"):
                try:
                    results = retriever._rrf_fusion(dense_res, sparse_res, top_k=top_k)
                except TypeError:
                    results = retriever._rrf_fusion(dense_res, sparse_res)[:top_k]
            elif hasattr(retriever, "_reciprocal_rank_fusion"):
                results = retriever._reciprocal_rank_fusion(dense_res, sparse_res, top_k=top_k)
            else:
                # Triển khai RRF chuẩn k=60 dự phòng nếu retriever không có hàm riêng
                rrf_scores = {}
                for rank, doc in enumerate(dense_res):
                    c_id, _, _ = _extract_doc_info(doc)
                    rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + 1.0 / (60 + rank + 1)
                for rank, doc in enumerate(sparse_res):
                    c_id, _, _ = _extract_doc_info(doc)
                    rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + 1.0 / (60 + rank + 1)

                doc_map = {_extract_doc_info(d)[0]: d for d in (dense_res + sparse_res)}
                sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
                results = [doc_map[cid] for cid in sorted_ids[:top_k] if cid in doc_map]

        # 4. Toàn bộ kiến trúc đề xuất: Hybrid RRF + Neo4j Graph Traversal Rerank
        else:
            results = retriever.retrieve(query, top_k=top_k)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Chuẩn hóa danh sách định danh để đối soát chính xác theo schema VietLawBench
        retrieved_chunks = []
        retrieved_docs = []
        contexts = []

        for r in results:
            c_id, doc_no, text = _extract_doc_info(r)
            if c_id:
                retrieved_chunks.append(c_id)
            if doc_no:
                retrieved_docs.append(str(doc_no).strip())
            if text:
                contexts.append(f"[{doc_no} - {c_id}]: {text}")

        output_payload = {
            "mode": mode,
            "top_k": top_k,
            "latency_ms": round(latency_ms, 2),
            "retrieved_chunks": retrieved_chunks,
            "retrieved_docs": retrieved_docs,
            "top_contexts": "\n\n".join(contexts[:top_k]),
            "results_count": len(results),
        }

        return {
            "output": json.dumps(output_payload, ensure_ascii=False),
        }

    except Exception as e:
        return {
            "error": f"Lỗi thực thi Retrieval Provider ({mode}): {str(e)}"
        }