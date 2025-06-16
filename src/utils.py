
import tempfile
import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def process_pdf(pdf_file, api_key):
    """
    Tải, chia nhỏ và tạo vector store từ một tệp PDF sử dụng API của OpenAI.
    """
    # Sử dụng tempfile để lưu tệp PDF tạm thời
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # 1. Tải tài liệu PDF
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # 2. Chia nhỏ văn bản thành các chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
        text_splitter = SemanticChunker(embeddings) # Semantic chunking

        splits = text_splitter.split_documents(docs)

        # 3. Tạo embedding model từ OpenAI
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # 4. Tạo vector store từ các chunks đã chia
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    finally:
        # Dọn dẹp tệp tạm
        os.remove(tmp_file_path)

    return vectorstore

def process_multiple_pdfs(pdf_files, openai_api_key):
    """
    Xử lý một danh sách các tệp PDF đã tải lên, trích xuất văn bản,
    chia nhỏ, tạo embeddings và tạo một vectorstore.
    """
    all_docs = []
    # Sử dụng tempfile để lưu trữ tạm thời các tệp PDF đã tải lên
    # vì PyPDFLoader yêu cầu đường dẫn tệp trên đĩa.
    with tempfile.TemporaryDirectory() as temp_dir:
        for pdf_file in pdf_files:
            temp_filepath = os.path.join(temp_dir, pdf_file.name)
            with open(temp_filepath, "wb") as f:
                f.write(pdf_file.getvalue())
            
            loader = PyPDFLoader(temp_filepath)
            docs = loader.load()
            all_docs.extend(docs)

    # Chia nhỏ văn bản từ tất cả các tài liệu
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # Tạo embeddings và vectorstore
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
    
    return vectorstore
    