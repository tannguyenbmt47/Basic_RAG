from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

def get_rag_chain(vectorstore, api_key):
    """
    Tạo một chuỗi RAG (Retrieval-Augmented Generation) từ vector store sử dụng OpenAI.
    """
    # Khởi tạo mô hình LLM từ OpenAI
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)

    # Tạo retriever từ vector store để tìm kiếm các chunks liên quan
    # retriever = vectorstore.as_retriever(search_type="similarity",
    #                                         search_kwargs={"k": 10})
    retriever = vectorstore.as_retriever(search_type="mmr",
                                    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7})
    

    # Tạo một prompt template để hướng dẫn LLM trả lời dựa trên context
    prompt_template_str = """
    Bạn là một trợ lý AI hữu ích chuyên trả lời các câu hỏi dựa trên một tài liệu được cung cấp.
    Hãy trả lời câu hỏi của người dùng chỉ dựa trên bối cảnh (context) sau đây.
    Nếu thông tin không có trong bối cảnh, hãy nói rằng bạn không tìm thấy câu trả lời trong tài liệu.
    Không được bịa đặt thông tin.

    Bối cảnh (Context):
    {context}

    Câu hỏi:
    {question}

    Trả lời (bằng tiếng Việt):
    """
    prompt = PromptTemplate.from_template(prompt_template_str)

    # Tạo chuỗi RAG bằng LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever