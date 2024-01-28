import streamlit as st
## streamlit에서 행한 행동이 로그로 남게 하기 위한 라이브러리
from loguru import logger

import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

## 토큰 개수를 세기위한 라이브러리
import tiktoken

from langchain_google_genai import ChatGoogleGenerativeAI

## 메모리를 가진 체인이 필요
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

## 여러 유형의 문서를 이해하기위한 라이브러리(PDF, DOC, PPT)
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

## 텍스트 splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
## 허깅페이스를 사용한 임베딩
from langchain.embeddings import HuggingFaceEmbeddings

## 벡터로 저장하기 위한 라이브러리
# from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate

## 몇개까지의 대화를 메모리를 넣어줄지 정하기
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory


# result = llm.invoke("gemini-pro를 활용한 챗봇 만드는 파이썬 코드 작성해줘")
# print(result.content)

## 스티림릿

# 메인 함수
def main():
    st.set_page_config(
        page_title="Gemini RAG",
        page_icon=":books:"
    )

    st.title("_Private Data :red[QA Chat]_ :books:")

    ## session_state를 쓰기 위해서 정의하는 함수
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    if "chat_disabled" not in st.session_state:
        st.session_state.chat_disabled = True

    if "retrieval" not in st.session_state:
        st.session_state.retrieval = None

    with st.sidebar:
        uploaded_files = st.file_uploader("UPload your file", type=["pdf", 'docx', 'pptx'], accept_multiple_files=True)
        gemini_api_key = st.text_input("GEMINI API Key", key="chatbot_api_key", type="password", help = "발급방법: https://godcode.tistory.com/35")
        # with st.expander("gemini api key 발급받는 방법"):
        #     st.markdown("gemini api key 발급받는 방법", help = "https://godcode.tistory.com/35")
        process = st.button("Process")

    if process:
        if not gemini_api_key:
            st.info("Please add your gemini API key to continue.")
            st.stop()
        with st.sidebar:
            with st.empty():
                with st.spinner("데이터 벡터화 중..."):
                    files_text = get_text(uploaded_files)
                    text_chunks = get_text_chunks(files_text)
                    vectorestore = get_vectorstore(text_chunks)
                    # st.session_state.retrieval = vectorestore


                st.success("데이터 벡터화 완료!")

        st.session_state.conversation = get_conversation_chain(gemini_api_key, vector_index=vectorestore)
        st.session_state.processComplete = True

        with st.sidebar:
            st.success("제미니 불러오기 완료!")
        
        st.session_state.chat_disabled = False

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role' : 'assistant',
                                         "content" : "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]
        
    for message in st.session_state.messages:
        ## 메시지마다 어떤 아이콘을 넣을지
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chat_history = StreamlitChatMessageHistory(key="chat_message")

    # Chat logic
    if st.session_state.chat_disabled:
        if query := st.chat_input("제미니가 도착할때까지 좀만 기다려주세요...", disabled=st.session_state.chat_disabled):
            st.session_state.messages.append({"role": "user",
                                          "content" : query})
        
            with st.chat_message("user"):
                st.markdown(query)
    else:
        if query := st.chat_input("질문을 입력해주세요.", disabled=st.session_state.chat_disabled):
            st.session_state.messages.append({"role": "user",
                                            "content" : query})
            
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):

                chain = st.session_state.conversation

                with st.spinner("답을 고민 중입니당..."):
                    # chain.run() 
                    result = chain({'retriever':st.session_state.retrieval, "query": query})
                    print(result)
                    # with get_openai_callback() as cb:
                    # st.session_state.chat_history = result['chat_history']
                    response = result['result']
                    source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(f"출처: {source_documents[0].metadata['source']}의 {source_documents[0].metadata['page']} 페이지", help=source_documents[0].page_content)          # help를 붙이면 ?아이콘 생김 -> 마우스 대면 원하는 글이 뜸
                    st.markdown(f"출처: {source_documents[1].metadata['source']}의 {source_documents[1].metadata['page']} 페이지", help=source_documents[1].page_content)
                    st.markdown(f"출처: {source_documents[2].metadata['source']}의 {source_documents[2].metadata['page']} 페이지", help=source_documents[2].page_content)
                    


    # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})



# 유틸리티 함수
        
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    return len(tokens)


def get_text(docs):

    doc_list = []

    for doc in docs:
        ## streamlit 서버 상에 파일이 업로드되면서 경로가 바뀌기에 서버상 경로로 설정
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")

        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()

        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()

        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)

    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):

    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # vectordb = FAISS.from_documents(text_chunks, embeddings)
    docsearch = Chroma.from_documents(text_chunks, hf).as_retriever(search_type="mmr", search_kwargs={'k':3, 'fetch_k': 10})

    return docsearch


def get_conversation_chain(gemini_api_key, vector_index):
    
    gemini = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=gemini_api_key, temperature=0.1, convert_system_message_to_human=True)

    # prompt

    # _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question in the original language, either Korean or English.

    # Chat History:
    # {chat_history}
    # Follow Up Input: {question}
    # Standalone question:"""
    # KOEN_CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    prompt_template = """Use the following context when answering the question. If you don't know the answer, reply to the "Following Text" in the header and answer to the best of your knowledge, or if you do know the answer, answer without the "Following Text". Answer question in its origin language, either Korean or English.
    
    Following Text: "주어진 정보에서 답변을 찾지는 못했지만, 제가 아는 선에서 답을 말씀드려볼게요! **틀릴 수도 있으니 교차검증은 필수입니다!**"
    
    following context: {context}

    Question: {question}
    Helpful Answer:"""
    KOEN_QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #         llm=gemini, 
    #         chain_type="stuff", 
    #         retriever=vectorstore.as_retriever(search_type="mmr", vervose=True),
    #         # condense_question_prompt=KOEN_QA_PROMPT,
    #         memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
    #         get_chat_history=lambda h: h,
    #         return_source_documents=True,
    #         verbose = True
    #     )

    # conversation_chain = load_qa_chain(
    #     llm=gemini, 
    #     chain_type="stuff", 
    #     prompt=KOEN_QA_PROMPT
    #     )
    
    
    conversation_chain = RetrievalQA.from_chain_type(
        gemini,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": KOEN_QA_PROMPT}
    )

    return conversation_chain



if __name__ == "__main__":
    main()
