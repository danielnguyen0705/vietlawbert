import httpx
import pytest
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from api.providers import MockProvider
from ui.client import ChatServiceError, ask
from ui.formatting import format_answer


def payload():
    return dict(MockProvider().ask("giao thông", 2), mode="mock", latency_ms=12.3)


def test_chainlit_loader_preserves_application_imports():
    # Reproduce the launcher's sys.path mutation from the documented ui/ cwd.
    subprocess.run([sys.executable, "-c", "from chainlit.config import load_module; load_module('chainlit_app.py'); from api.schemas import ChatResponse; from ui.client import ask"],
                   cwd=Path(__file__).resolve().parents[2] / "ui", check=True)


@pytest.mark.asyncio
async def test_client_and_rendering():
    def handler(request):
        assert request.url.path == "/api/v1/chat"
        return httpx.Response(200, json=payload())
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        data = await ask(client, "giao thông", 2)
    text, sources = format_answer(data, "turn1")
    other, other_sources = format_answer(data, "turn2")
    assert "100.0%" in text and "12 ms" in text and "minh họa" in text
    assert sources[0][0] in text and sources[0][0] not in other
    assert sources[0][0] != other_sources[0][0]
    data["attribution_score"] = None
    data["contexts"][0]["graph_boost"] = None
    text, sources = format_answer(data, "turn3")
    assert "Chưa có" in text and "Chưa có" in sources[0][1]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,expected", [
    (httpx.ConnectError("connection"), "Không kết nối"),
    (httpx.ReadTimeout("timeout"), "quá thời gian"),
    (503, "đang bận"), (502, "tạo câu trả lời"), (422, "không hợp lệ"),
    (500, "lỗi xử lý"), (200, "không đúng định dạng"),
])
async def test_errors(failure, expected):
    def handler(request):
        if isinstance(failure, Exception):
            raise failure
        return httpx.Response(failure, json={})
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ChatServiceError, match=expected):
            await ask(client, "x", 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, httpx.ConnectError("offline"), httpx.ReadTimeout("slow")])
async def test_callback_finishes_pending_message(monkeypatch, failure):
    monkeypatch.chdir(Path(__file__).resolve().parents[2] / "ui")
    from ui import chainlit_app as app

    emitted = []
    class Message:
        def __init__(self, content):
            self.content, self.elements, self.updated = content, [], False
            emitted.append(self)
        async def send(self):
            pass
        async def update(self):
            self.updated = True

    def handler(request):
        if failure:
            raise failure
        return httpx.Response(200, json=payload())

    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        state = {"client": client, "chat_settings": {"top_k": 2}}
        sidebar = AsyncMock()
        monkeypatch.setattr(app.cl, "user_session", SimpleNamespace(get=state.get, set=state.__setitem__))
        monkeypatch.setattr(app.cl, "Message", Message)
        monkeypatch.setattr(app.cl, "Text", lambda **kwargs: kwargs)
        monkeypatch.setattr(app.cl, "ElementSidebar", SimpleNamespace(set_elements=sidebar))
        await app.message(SimpleNamespace(content="Demo giao thông"))
        assert emitted[0].updated
        assert "Đang xử lý" not in emitted[0].content
        assert len(emitted[0].elements) == (0 if failure else 2)
        sidebar.assert_awaited_once_with([])
