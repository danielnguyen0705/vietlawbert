import httpx
from api.schemas import ChatResponse


class ChatServiceError(Exception):
    pass


async def ask(client: httpx.AsyncClient, query: str, top_k: int):
    try:
        response = await client.post("/api/v1/chat", json={"query": query, "top_k": top_k, "stream": False})
        response.raise_for_status()
        data = response.json()
        # Shared wire contract is lightweight and never imports the RAG stack.
        return ChatResponse.model_validate(data).model_dump()
    except httpx.TimeoutException as exc:
        raise ChatServiceError("Backend phản hồi quá thời gian chờ. Vui lòng gửi lại sau.") from exc
    except httpx.RequestError as exc:
        raise ChatServiceError("Không kết nối được backend. Kiểm tra FastAPI và gửi lại câu hỏi.") from exc
    except httpx.HTTPStatusError as exc:
        messages = {422: "Câu hỏi hoặc số căn cứ không hợp lệ.", 503: "Backend chưa sẵn sàng hoặc đang bận.", 502: "Dịch vụ tạo câu trả lời đang gặp lỗi."}
        raise ChatServiceError(messages.get(exc.response.status_code, "Backend gặp lỗi xử lý.") + " Vui lòng gửi lại sau.") from exc
    except (ValueError, TypeError) as exc:
        raise ChatServiceError("Phản hồi backend không đúng định dạng. Vui lòng gửi lại sau.") from exc
