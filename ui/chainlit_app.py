import os
import logging
import sys
from pathlib import Path

# Chainlit is intentionally launched with ui/ as its working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import chainlit as cl
import httpx
from chainlit.input_widget import Slider
from dotenv import load_dotenv

from ui.client import ChatServiceError, ask
from ui.formatting import format_answer

load_dotenv(ROOT / ".env")


@cl.set_starters
async def starters():
    if os.getenv("CHAT_MODE") == "rag":
        return []
    return [cl.Starter(label=label, message=f"Demo {label.lower()}")
            for label in ("Giao thông", "Doanh nghiệp", "Hành chính")]


@cl.on_chat_start
async def start():
    client = httpx.AsyncClient(
        base_url=os.getenv("CHAT_API_URL", "http://localhost:8000"),
        timeout=httpx.Timeout(float(os.getenv("CHAT_TIMEOUT_SECONDS", "300")), connect=5.0),
    )
    cl.user_session.set("client", client)
    await cl.ChatSettings([Slider(id="top_k", label="Số căn cứ tối đa", initial=3, min=1, max=10, step=1)]).send()
    try:
        response = await client.get("/api/v1/health", timeout=5.0)
        response.raise_for_status()
        mode = response.json()["mode"]
        greeting = "**Chế độ demo — dữ liệu minh họa.**" if mode == "mock" else "**Chế độ RAG.**"
    except (httpx.HTTPError, ValueError, KeyError):
        greeting = "Backend chưa sẵn sàng. Bạn có thể gửi lại câu hỏi sau khi backend khởi động."
    hint = "Mỗi câu hỏi được xử lý độc lập. Hãy hỏi về số hiệu và điều khoản trong tập văn bản đã nạp."
    if 'mode' in locals() and mode == "mock":
        hint = "Mỗi câu hỏi được xử lý độc lập. Thử: Demo giao thông, Demo doanh nghiệp hoặc Demo hành chính."
    await cl.Message(content=greeting + "\n\n" + hint).send()


@cl.on_settings_update
async def settings_update(settings):
    cl.user_session.set("chat_settings", settings)


@cl.on_message
async def message(message: cl.Message):
    turn = cl.user_session.get("turn_count", 0) + 1
    cl.user_session.set("turn_count", turn)
    pending = cl.Message(content="Đang xử lý câu hỏi…")
    await pending.send()
    try:
        client = cl.user_session.get("client")
        if client is None:
            raise ChatServiceError("Phiên kết nối đã kết thúc. Vui lòng mở cuộc trò chuyện mới.")
        settings = cl.user_session.get("chat_settings") or {}
        data = await ask(client, message.content, int(settings.get("top_k", 3)))
        pending.content, sources = format_answer(data, turn)
        pending.elements = [cl.Text(name=name, content=content, display="side") for name, content in sources]
    except ChatServiceError as exc:
        pending.content = str(exc)
    except Exception:
        logging.getLogger(__name__).exception("chat_ui_failed")
        pending.content = "Có lỗi hiển thị. Vui lòng mở cuộc trò chuyện mới và gửi lại."
    await pending.update()
    # Chainlit may auto-append newly sent side elements to the open panel.
    # Close it after each response; a source link opens exactly that source.
    await cl.ElementSidebar.set_elements([])


@cl.on_chat_end
async def end():
    client = cl.user_session.get("client")
    if client is not None:
        await client.aclose()
