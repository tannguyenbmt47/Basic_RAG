import streamlit as st
import tempfile
import os
from dotenv import load_dotenv # Thêm import này
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

from src.rag import get_rag_chain
from src.utils import process_pdf, process_multiple_pdfs
# Tải các biến môi trường từ tệp .env
load_dotenv()
# --- GIAO DIỆN STREAMLIT ---

st.set_page_config(page_title="Retrieval Chatbot", layout="wide")

st.title("")
st.write("Tải lên một tài liệu PDF và đặt câu hỏi về nội dung của nó.")

# Lấy API key từ biến môi trường
openai_api_key = os.getenv("OPENAI_API_KEY")

# Khởi tạo session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state: # THÊM DÒNG NÀY
    st.session_state.retriever = None

# Sidebar để tải lên tệp PDF
with st.sidebar:
    st.logo("img/text.png", size = 'large',icon_image="img/logo_no_bg.png")
    st.header("Tài liệu của bạn")
    

    pdf_files = st.file_uploader("Tải lên tệp PDF của bạn", type="pdf", accept_multiple_files=True)

    if st.button("Xử lý tài liệu"):
        if not openai_api_key:
            st.error("Không thể xử lý: Thiếu OpenAI API Key.")
        elif not pdf_files:
            st.error("Vui lòng tải lên một tệp PDF.")
        else:
            with st.spinner("Đang xử lý tài liệu... Vui lòng đợi."):
                try:
                    # Lưu trữ vectorstore và RAG chain vào session state
                    st.session_state.vectorstore = process_multiple_pdfs(pdf_files, openai_api_key)
                    st.session_state.rag_chain, st.session_state.retriever = get_rag_chain(st.session_state.vectorstore, openai_api_key)
                    st.success("Tài liệu đã được xử lý xong! Bây giờ bạn có thể đặt câu hỏi.")
                    # Xóa tin nhắn cũ khi có tài liệu mới
                    st.session_state.messages = []
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {e}")

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận input từ người dùng
if user_question := st.chat_input("Đặt câu hỏi về tài liệu của bạn..."):
    # Thêm câu hỏi của người dùng vào lịch sử chat
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Kiểm tra xem API key và tài liệu đã sẵn sàng chưa
    if not openai_api_key:
        st.warning("Không thể trả lời: Vui lòng cấu hình OpenAI API Key trong tệp .env.")
    elif st.session_state.rag_chain is None:
        st.warning("Vui lòng tải lên và xử lý một tài liệu PDF trước.")
    else:
        # Nếu đã sẵn sàng, gọi RAG chain để lấy câu trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                try:
                    # Sử dụng `stream` để có hiệu ứng gõ chữ
                    if st.session_state.retriever:
                        retrieved_docs = st.session_state.retriever.invoke(user_question)
                        st.info(f"Đã truy xuất {len(retrieved_docs)} tài liệu liên quan:")
                        for i, doc in enumerate(retrieved_docs):
                            with st.expander(f"Tài liệu truy xuất #{i+1} (Page: {doc.metadata.get('page', 'N/A')})"):
                                st.write(doc.page_content)
                    response_stream = st.session_state.rag_chain.stream(user_question)
                    full_response = st.write_stream(response_stream)
                    # Thêm câu trả lời của trợ lý vào lịch sử chat
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi khi tạo câu trả lời: {e}")
