from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"query": "Demo giao thông", "top_k": 3, "stream": False}]})
    query: str = Field(min_length=1, max_length=4000)
    top_k: StrictInt = Field(default=3, ge=1, le=10)
    stream: Literal[False] = False

    @field_validator("query", mode="before")
    @classmethod
    def trim_query(cls, value):
        return value.strip() if isinstance(value, str) else value


class LawContext(BaseModel):
    chunk_id: str | None = None
    doc_number: str | None = None
    hierarchy_path: str | None = None
    content: str
    final_rerank_score: float | None = None
    graph_boost: float | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "query": "Demo giao thông", "answer": "Dữ liệu minh họa, không phải tư vấn pháp lý.",
        "contexts": [], "has_context": False, "attribution_score": 0.0,
        "latency_ms": 1.2, "model_used": "mock-fixture", "mode": "mock",
    }]})
    query: str
    answer: str
    contexts: list[LawContext]
    has_context: bool
    attribution_score: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float = Field(ge=0)
    model_used: str | None = None
    mode: Literal["mock", "rag"]


class HealthResponse(BaseModel):
    mode: Literal["mock", "rag"]
    ready: bool
    provider: Literal["ready", "unavailable"]
    model: Literal["not_required", "loaded", "unavailable"]
    databases: dict[str, str]


class ErrorResponse(BaseModel):
    detail: str
