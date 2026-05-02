import streamlit as st
import os
import time
from src.rag.generator import LegalGenerator

# ==========================================
# 1. HÀM ĐỔI THEME GỐC CỦA STREAMLIT
# ==========================================
def switch_native_theme(mode):
    """
    Hàm này can thiệp trực tiếp vào file config của Streamlit
    để ép toàn bộ hệ thống đổi màu chuẩn xác 100% không bị sót viền.
    """
    os.makedirs(".streamlit", exist_ok=True)
    config_path = ".streamlit/config.toml"
    
    with open(config_path, "w", encoding="utf-8") as f:
        if mode == "light":
            # Giao diện sáng gốc
            f.write('[theme]\nbase="light"\nprimaryColor="#1976D2"\n')
        else:
            # Giao diện tối gốc
            f.write('[theme]\nbase="dark"\nprimaryColor="#81C784"\n')
    
    # Lưu trạng thái, chờ 0.3s cho file lưu xong rồi F5 lại app
    st.session_state.theme = mode
    time.sleep(0.3)
    st.rerun()

# Khởi tạo state mặc định (nếu chưa có)
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ==========================================
# 2. CẤU HÌNH TRANG
# ==========================================
st.set_page_config(page_title="VietLawBERT AI", page_icon="⚖️", layout="wide")

# ==========================================
# 3. SIDEBAR (CÔNG TẮC ĐIỀU KHIỂN)
# ==========================================
with st.sidebar:
    st.title("⚖️ VietLawBERT")
    st.markdown("---")
    st.write("### Giao diện hệ thống")
    
    # Hai nút bấm song song
    col1, col2 = st.columns(2)
    if col1.button("☀️ Sáng", use_container_width=True):
        if st.session_state.theme != "light":
            switch_native_theme("light")
            
    if col2.button("🌙 Tối", use_container_width=True):
        if st.session_state.theme != "dark":
            switch_native_theme("dark")
            
    st.markdown("---")
    st.info("**Phiên bản:** v2.5 (Native Theme)\n\n**Core:** Gemini 2.5 Flash\n\n**Chế độ:** Hybrid RAG")

# ==========================================
# 4. MAKEUP RIÊNG CHO BONG BÓNG CHAT
# ==========================================
# Giao diện đã chuẩn, giờ chỉ cần tô màu bong bóng chat cho hợp tone User/Bot
if st.session_state.theme == "light":
    bubble_css = """
    <style>
    [data-testid="stChatMessageUser"] {
        background-color: #E3F2FD !important;
        border-right: 4px solid #1976D2;
        border-radius: 20px 20px 5px 20px;
        margin-left: 20%;
    }
    [data-testid="stChatMessageAssistant"] {
        background-color: #E8F5E9 !important;
        border-left: 4px solid #2E7D32;
        border-radius: 20px 20px 20px 5px;
        margin-right: 20%;
    }
    </style>
    """
else:
    bubble_css = """
    <style>
    [data-testid="stChatMessageUser"] {
        background-color: #1A237E !important;
        border-right: 4px solid #64B5F6;
        border-radius: 20px 20px 5px 20px;
        margin-left: 20%;
    }
    [data-testid="stChatMessageAssistant"] {
        background-color: #1B5E20 !important;
        border-left: 4px solid #81C784;
        border-radius: 20px 20px 20px 5px;
        margin-right: 20%;
    }
    </style>
    """
st.markdown(bubble_css, unsafe_allow_html=True)

# ==========================================
# 5. KHỞI TẠO VÀ CHẠY BOT
# ==========================================
st.title("⚖️ HỆ THỐNG HỎI ĐÁP LUẬT GIAO THÔNG ĐƯỜNG BỘ")
st.markdown("Hệ thống **trợ lý ảo pháp lý** tư vấn Luật Giao thông Đường bộ Việt Nam.")

@st.cache_resource
def get_bot():
    return LegalGenerator()

bot = get_bot()

# Nạp lịch sử
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Xin chào! Tôi là trợ lý pháp lý **VietLawBERT**.\n\nBạn cần tra cứu quy định hay mức phạt nào về giao thông hôm nay?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Khung nhập liệu (Hỗ trợ Enter mặc định)
if prompt := st.chat_input("Nhập tình huống vi phạm (Ví dụ: Vượt đèn đỏ bị phạt bao nhiêu?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Đang tra cứu cơ sở dữ liệu luật..."):
            try:
                response = bot.ask(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ Hệ thống gặp sự cố: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})