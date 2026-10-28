"""
demo/server/feedback_service.py — Microservice: RLHF Feedback
==============================================================
Ghi like/dislike của người dùng vào JSONL để phục vụ:
  - Fine-tuning RLHF sau này
  - Phân tích chất lượng câu trả lời (RQ4)

File: flagging/feedback.jsonl (gitignored — thêm vào .gitignore)
Schema mỗi dòng:
{
    "timestamp"  : "2025-01-15T10:30:00.000Z",
    "session_id" : "abc123",
    "query"      : "...",
    "answer"     : "...",
    "liked"      : true/false,
    "index"      : 2,
    "mode"       : "Full (+ Graph Rerank)",
    "top_k"      : 5,
    "model_used" : "Primary_Cloud_SOTA"
}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from demo.config import FLAGGING_DIR, FEEDBACK_FILE

logger = logging.getLogger("demo.feedback_service")


def _ensure_flagging_dir() -> None:
    FLAGGING_DIR.mkdir(parents=True, exist_ok=True)


def log_feedback(
    query: str,
    answer: str,
    liked: bool,
    index: int,
    mode: str = "",
    top_k: int = 5,
    model_used: str = "",
    session_id: Optional[str] = None,
) -> bool:
    """
    Ghi một bản ghi feedback vào FEEDBACK_FILE.

    Returns
    -------
    bool — True nếu ghi thành công, False nếu có lỗi.
    """
    _ensure_flagging_dir()

    record = {
        "timestamp"  : datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "session_id" : session_id or "",
        "query"      : query,
        "answer"     : answer,
        "liked"      : liked,
        "index"      : index,
        "mode"       : mode,
        "top_k"      : top_k,
        "model_used" : model_used,
    }

    try:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Feedback ghi nhận: index=%d | liked=%s | query='%.40s...'",
            index, liked, query,
        )
        return True
    except Exception as exc:
        logger.error("Không thể ghi feedback: %s", exc)
        return False


def get_feedback_stats() -> dict:
    """Thống kê nhanh số liệu feedback hiện có."""
    if not FEEDBACK_FILE.exists():
        return {"total": 0, "liked": 0, "disliked": 0}

    total = liked = 0
    try:
        with open(FEEDBACK_FILE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    total += 1
                    if rec.get("liked"):
                        liked += 1
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.warning("Không đọc được feedback file: %s", exc)
        return {"total": 0, "liked": 0, "disliked": 0}

    return {"total": total, "liked": liked, "disliked": total - liked}
