from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class HarnessConfig:
    """Central thresholds for lightweight CI-safe validation."""

    minimum_content_chars: int = 120
    minimum_vietnamese_score: float = 0.25
    maximum_duplicate_ratio: float = 0.01
    require_source_url: bool = True
    require_document_id: bool = True
    require_title: bool = False

    @classmethod
    def from_env(cls) -> "HarnessConfig":
        return cls(
            minimum_content_chars=int(os.getenv("VIETLAW_MIN_CONTENT_CHARS", "120")),
            minimum_vietnamese_score=float(os.getenv("VIETLAW_MIN_VI_SCORE", "0.25")),
            maximum_duplicate_ratio=float(os.getenv("VIETLAW_MAX_DUP_RATIO", "0.01")),
            require_source_url=os.getenv("VIETLAW_REQUIRE_SOURCE_URL", "1") != "0",
            require_document_id=os.getenv("VIETLAW_REQUIRE_DOCUMENT_ID", "1") != "0",
            require_title=os.getenv("VIETLAW_REQUIRE_TITLE", "0") == "1",
        )
