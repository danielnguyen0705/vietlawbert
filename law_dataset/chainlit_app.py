"""
chainlit_app.py - Giao diện chat VietLawBERT sử dụng Chainlit.

Cấu trúc:
1. Khởi tạo LegalGenerator (cache trong user session).
2. Gửi tin nhắn chào mừng khi bắt đầu phiên chat.
3. Xử lý câu hỏi từ người dùng với loading indicator.
4. Xử lý lỗi khi backend AI chưa sẵn sàng.
"""

import os
import sys
import logging

import chainlit as cl

from rag.generator import LegalGenerator

# ==========================================
# 1. CẤU HÌNH LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
)
logger = logging.getLogger("ChainlitApp")


# ==========================================
# 2. KHỞI TẠO CHAT SESSION
# ==========================================
@cl.on_chat_start
async def start_chat():
    """Khởi tạo session chat, load bot và gửi lời chào."""
    try:
        # Khởi tạo LegalGenerator và lưu vào user session
        bot = LegalGenerator()
        cl.user_session.set("bot", bot)

        # Lưu thông tin version/model cho hiển thị
        cl.user_session.set(
            "model_info",
            {
                "version": "v2.5",
                "core": os.getenv("GENERATOR_MODEL", "deepseek-r1:14b"),
                "mode": "Hybrid RAG (Milvus + Neo4j)",
            },
        )

        # Gửi tin nhắn chào mừng
        welcome_msg = (
            "👋 Xin chào! Tôi là trợ lý pháp lý **VietLawBERT**.\n\n"
            "Bạn cần tra cứu quy định hay mức phạt nào về giao thông hôm nay?"
        )
        await cl.Message(content=welcome_msg, author="VietLawBERT").send()

    except Exception as e:
        logger.error(f"Không thể khởi tạo Generator: {e}")
        error_msg = (
            "❌ **Backend AI chưa sẵn sàng!** Vui lòng kiểm tra:\n\n"
            "1. Ollama đang chạy: `ollama serve`\n"
            f"2. Model đã tải: `ollama pull {os.getenv('GENERATOR_MODEL', 'deepseek-r1:14b')}`\n"
            "3. Kiểm tra `.env` có đúng `OLLAMA_BASE_URL`\n\n"
            f"**Chi tiết lỗi:** {str(e)}"
        )
        await cl.Message(content=error_msg, author="System").send()


# ==========================================
# 3. XỬ LÝ TIN NHẮN NGƯỜI DÙNG
# ==========================================
@cl.on_message
async def main(message: cl.Message):
    """Xử lý câu hỏi từ người dùng."""
    bot = cl.user_session.get("bot")

    # Kiểm tra nếu bot chưa được khởi tạo
    if bot is None:
        await cl.Message(
            content="❌ Bot chưa được khởi tạo. Vui lòng kết nối lại phiên chat.",
            author="System",
        ).send()
        return

    user_query = message.content

    # Tạo tin nhắn response với loading indicator
    msg = cl.Message(content="", author="VietLawBERT")
    await msg.send()

    try:
        # Stream từng chunk khi bot trả về kết quả
        # Nếu bot.ask() hỗ trợ streaming thì dùng async generator
        # Nếu không, gọi đồng bộ và hiển thị toàn bộ
        response = await _call_bot_async(bot, user_query)

        # Cập nhật nội dung tin nhắn với response
        msg.content = response
        await msg.update()

    except Exception as e:
        error_msg = f"❌ Hệ thống gặp sự cố: {str(e)}"
        msg.content = error_msg
        await msg.update()
        logger.error(f"Lỗi khi xử lý câu hỏi: {e}")


# ==========================================
# 4. HÀM HỖ TRỢ GỌI BOT BẤT ĐỒNG BỘ
# ==========================================
async def _call_bot_async(bot, query: str) -> str:
    """
    Wrapper bất đồng bộ cho bot.ask().
    Chạy blocking call trong thread pool để không block event loop.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, bot.ask, query)
    return response


# ==========================================
# 5. XỬ LÝ KHI CHAT KẾT THÚC
# ==========================================
@cl.on_chat_end
def end_chat():
    """Cleanup khi phiên chat kết thúc."""
    logger.info("Phiên chat kết thúc.")