"""
test_chatbox_handlers.py — Kiểm thử logic hội thoại của Gradio chatbox (demo/ui/handlers.py).

Phạm vi: lớp XỬ LÝ SỰ KIỆN (không kiểm giao diện/DOM, không kiểm chất lượng pháp lý).
Chạy bằng backend MOCK → không cần Qdrant / Elasticsearch / Neo4j / LLM, nhanh và tất định.
Log (interactions.jsonl, feedback.jsonl) được chuyển vào thư mục tạm → KHÔNG ghi đè dữ liệu thật ở flagging/.

Tiêu chí kiểm (mỗi test = một bất biến của chatbox):
  1. Đầu vào rỗng / toàn khoảng trắng bị bỏ qua, không ghi log.
  2. Lịch sử đúng định dạng Gradio 6.x: [{"role","content"}, ...], user rồi assistant.
  3. Mỗi lượt ghi đúng 1 bản ghi interaction, turn_index tăng dần, khớp số evidence.
  4. top_k được tôn trọng; mọi chế độ trong dropdown đều chạy được.
  5. Log khớp 100% với bảng Evidence hiển thị trên UI.
  6. Clear xoá sạch trạng thái; Retry THAY lượt cuối (không nhân đôi lịch sử).
  7. Like/Dislike ánh xạ đúng message-index → lượt hỏi-đáp; chỉ số ngoài phạm vi bị bỏ qua.

Chạy:
    pytest tests/system/test/chatbox/test_chatbox_handlers.py -v
    pytest -m chatbox -v
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.tier5, pytest.mark.chatbox]

pytest.importorskip("gradio", reason="Cần `pip install gradio` để import demo.ui.handlers")


# ─── Fixture ────────────────────────────────────────────────────────────────

class Chat:
    """Bọc trạng thái (history, meta) như gr.State để test gọi handler giống UI."""

    def __init__(self, handlers, top_k=5, mode="Hybrid-RRF"):
        self.h, self.top_k, self.mode = handlers, top_k, mode
        self.history, self.meta = [], []
        self.evidence_html = self.metrics_html = ""

    def submit(self, message, top_k=None, mode=None):
        out = self.h.handle_submit(message, self.history, self.meta,
                                   mode or self.mode, self.top_k if top_k is None else top_k)
        _, self.history, self.meta, _, self.evidence_html, self.metrics_html = out
        return out

    def retry(self):
        out = self.h.handle_retry(self.history, self.meta, self.mode, self.top_k)
        _, self.history, self.meta, _, self.evidence_html, self.metrics_html = out
        return out

    def clear(self):
        _, self.history, self.meta, self.evidence_html, self.metrics_html = self.h.handle_clear()

    def like(self, msg_index, liked):
        self.h.handle_feedback(SimpleNamespace(index=msg_index, liked=liked), self.history, self.meta)


@pytest.fixture
def chat(tmp_path, monkeypatch):
    from demo.ui import handlers
    from demo.server import interaction_service as isv, feedback_service as fsv

    monkeypatch.setattr(handlers, "USE_MOCK_BACKEND", True)
    monkeypatch.setattr(isv, "FLAGGING_DIR", tmp_path)
    monkeypatch.setattr(isv, "INTERACTIONS_FILE", tmp_path / "interactions.jsonl")
    monkeypatch.setattr(fsv, "FLAGGING_DIR", tmp_path)
    monkeypatch.setattr(fsv, "FEEDBACK_FILE", tmp_path / "feedback.jsonl")

    c = Chat(handlers)
    c.interactions_file = tmp_path / "interactions.jsonl"
    c.feedback_file = tmp_path / "feedback.jsonl"
    return c


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ─── 1. Đầu vào rỗng ────────────────────────────────────────────────────────

@pytest.mark.parametrize("message", ["", "   ", "\n\t "])
def test_empty_message_is_ignored(chat, message):
    chat.submit(message)
    assert chat.history == [] and chat.meta == []
    assert _read_jsonl(chat.interactions_file) == []


# ─── 2. Định dạng lịch sử ───────────────────────────────────────────────────

def test_history_uses_gradio6_role_content_format(chat):
    chat.submit("  Điều kiện đăng ký doanh nghiệp?  ")
    assert [m["role"] for m in chat.history] == ["user", "assistant"]
    assert chat.history[0]["content"] == "Điều kiện đăng ký doanh nghiệp?"   # đã strip
    assert chat.history[1]["content"].strip() != ""


# ─── 3. Nhật ký mỗi lượt ────────────────────────────────────────────────────

def test_each_turn_writes_one_log_record_with_increasing_turn_index(chat):
    chat.submit("Câu hỏi 1")
    chat.submit("Câu hỏi 2")
    recs = _read_jsonl(chat.interactions_file)
    assert [r["turn_index"] for r in recs] == [0, 1]
    assert [r["query"] for r in recs] == ["Câu hỏi 1", "Câu hỏi 2"]
    assert len({r["run_id"] for r in recs}) == 1          # cùng một phiên chạy
    for r in recs:
        assert r["metrics"]["n_docs"] == len(r["evidence"])
        assert r["is_mock"] is True


# ─── 4. top_k & mode ────────────────────────────────────────────────────────

@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_top_k_is_respected(chat, top_k):
    chat.submit("Hồ sơ đăng ký doanh nghiệp gồm gì?", top_k=top_k)
    rec = _read_jsonl(chat.interactions_file)[-1]
    assert rec["top_k"] == top_k
    assert rec["metrics"]["n_docs"] == min(top_k, 3)       # mock backend có 3 doc


def test_every_dropdown_mode_runs_and_is_logged(chat):
    from demo.config import RETRIEVAL_MODES
    for mode in RETRIEVAL_MODES:
        chat.submit(f"Câu hỏi thử chế độ {mode}", mode=mode)
    recs = _read_jsonl(chat.interactions_file)
    assert [r["mode"] for r in recs] == list(RETRIEVAL_MODES)
    assert all(r["answer"].strip() for r in recs)


@pytest.mark.xfail(strict=False, reason="handler chưa kẹp top_k vào [1, MAX_TOP_K]; API /chat có thể nhận giá trị ngoài slider")
@pytest.mark.parametrize("top_k", [0, -1, 10_000])
def test_top_k_out_of_range_is_clamped(chat, top_k):
    from demo.config import MAX_TOP_K
    chat.submit("Câu hỏi biên", top_k=top_k)
    rec = _read_jsonl(chat.interactions_file)[-1]
    assert 1 <= rec["top_k"] <= MAX_TOP_K


# ─── 5. Log khớp UI ─────────────────────────────────────────────────────────

def test_logged_evidence_matches_ui_evidence_panel(chat):
    chat.submit("Quyền kinh doanh của doanh nghiệp là gì?", top_k=3)
    rec = _read_jsonl(chat.interactions_file)[-1]
    assert rec["evidence"], "mock backend phải trả về evidence"
    for ev in rec["evidence"]:
        assert ev["doc_number"] in chat.evidence_html
    assert [e["rank"] for e in rec["evidence"]] == list(range(1, len(rec["evidence"]) + 1))


# ─── 6. Clear / Retry ───────────────────────────────────────────────────────

def test_clear_resets_all_state(chat):
    from demo.ui.panels import render_evidence, render_metrics
    chat.submit("Câu hỏi 1")
    chat.clear()
    assert chat.history == [] and chat.meta == []
    assert chat.evidence_html == render_evidence([])      # panel Evidence về trạng thái rỗng
    assert chat.metrics_html == render_metrics()          # panel Metrics về mặc định


def test_retry_replaces_last_turn_instead_of_appending(chat):
    chat.submit("Câu hỏi 1")
    chat.submit("Câu hỏi 2")
    chat.retry()
    assert len(chat.history) == 4 and len(chat.meta) == 2          # không phình lịch sử
    assert chat.history[-2]["content"] == "Câu hỏi 2"             # đúng câu hỏi cuối
    last = _read_jsonl(chat.interactions_file)[-1]
    assert last["query"] == "Câu hỏi 2" and last["turn_index"] == 1


def test_retry_on_empty_history_is_noop(chat):
    chat.retry()
    assert chat.history == [] and _read_jsonl(chat.interactions_file) == []


# ─── 7. Like / Dislike ──────────────────────────────────────────────────────

def test_feedback_maps_message_index_to_turn(chat):
    chat.submit("Câu hỏi 1")
    chat.submit("Câu hỏi 2")
    chat.like(msg_index=3, liked=False)       # message #3 = câu trả lời của lượt 2 (flat list user0,asst0,user1,asst1)
    fb = _read_jsonl(chat.feedback_file)
    assert len(fb) == 1
    assert fb[0]["query"] == "Câu hỏi 2" and fb[0]["index"] == 1 and fb[0]["liked"] is False
    assert fb[0]["answer"] == chat.history[3]["content"]


def test_feedback_out_of_range_is_ignored(chat):
    chat.submit("Câu hỏi 1")
    chat.like(msg_index=99, liked=True)
    assert _read_jsonl(chat.feedback_file) == []
