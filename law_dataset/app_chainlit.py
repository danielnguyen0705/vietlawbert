"""
app_chainlit.py - Giao diện Chatbot chuyên nghiệp sử dụng Chainlit.
Hỗ trợ:
  - Hiển thị các bước suy nghĩ (Steps Reasoning)
  - Streaming câu trả lời từ LLM thời gian thực
  - Kết nối Hybrid RAG (Milvus + Neo4j)
"""

import os
import sys
import chainlit as cl

# Thêm src vào PYTHONPATH để load module rag
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from config import config
from rag.retriever import LegalRetriever
from openai import OpenAI

# Khởi tạo OpenAI client cho giao diện Chainlit
client = OpenAI(base_url=config.LLM_API_BASE, api_key=config.LLM_API_KEY)

@cl.on_chat_start
async def start():
    # Khởi tạo retriever và lưu vào session
    retriever = LegalRetriever()
    cl.user_session.set("retriever", retriever)

    # Lời chào đầu tiên
    await cl.Message(
        content="👋 Xin chào! Tôi là trợ lý pháp lý chuyên sâu **VietLawBERT**.\n\nBạn cần tra cứu quy định hay mức phạt nào về giao thông hôm nay?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Lấy retriever từ session
    retriever = cl.user_session.get("retriever")
    query = message.content

    # TẠO STEP 1: Truy xuất tài liệu (Retrieval Step)
    retrieval_step = cl.Step(name="Hybrid RAG Search (Milvus & Neo4j)")
    await retrieval_step.send()

    contexts = retriever.search_context(query)

    if not contexts:
        retrieval_step.output = "Không tìm thấy tài liệu liên quan."
        await retrieval_step.update()
        await cl.Message(content="Dữ liệu hiện tại không đề cập, tôi chỉ chuyên về luật giao thông và không thể trả lời câu hỏi đó.").send()
        return

    # Show tài liệu đã tìm thấy trong step
    doc_summary = ""
    for idx, c in enumerate(contexts):
        doc_summary += f"Tài liệu {idx+1}: {c['doc_info']} | {c['article']} | Hiệu lực: {c['effective_date']}\n"
    retrieval_step.output = doc_summary
    await retrieval_step.update()

    # Xây dựng prompt
    context_str = ""
    for i, c in enumerate(contexts):
        context_str += f"TAI LIEU {i+1}:\n- Van ban: {c['doc_info']}\n- Hieu luc: {c['effective_date']}\n- Dieu: {c['article']}\n- Noi dung: {c['content']}\n\n"

    prompt = f"""
Ban la VietLawBERT, mot chuyen gia ve luat giao thong tai Viet Nam.
NHIEM VU: Dựa trên cac tai lieu luat giao thong da duoc cung cap trong phan CONTEXT, hay tra loi cau hoi cua nguoi dung mot cach chinh xac va ngan gon nhat co the.

CAC QUY TAC BAT BUOC:
1. Bat dau cau tra loi bang cau truc: "Theo [Ten van ban/So hieu] (co hieu luc tu [Ngay hieu luc]), ...", Sau do moi di vao noi dung cau tra loi ngan gon.
2. TUYET DOI khong duoc bia them thong tin.
3. Neu context khong co thong tin, hay bao la "Du lieu hien tai khong de cap, toi chi chuyen ve luat giao thong va khong thể tra loi cau hoi do".

CONTEXT:
{context_str}

CAU HOI: {query}
"""

    # TẠO STEP 2: Sinh câu trả lời (LLM Generation Step) với cơ chế Streaming
    msg = cl.Message(content="")

    # Gọi OpenAI API dạng stream
    stream = client.chat.completions.create(
        model=config.GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
        stream=True
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            await msg.stream_token(token)

    await msg.send()
