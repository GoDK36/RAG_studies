import pprint, os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.chains import create_extraction_chain
# from langchain.chat_models import ChatOpenAI
# from langchain_community import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_community.llms import huggingface_pipeline
from langchain_community.llms import huggingface_hub


def scrape_with_playwright(urls, extraction_chain):
    # URL 로부터 본문 내용 웹스크래핑
    loader = AsyncChromiumLoader(urls)
    # 웹스크래핑 내용 로드
    docs = loader.load()
    # HTML 로더로 컨텐츠 로드
    bs_transformer = BeautifulSoupTransformer()
    # 특정 태그에 대한 내용만 추출 ('span')
    docs_transformed = bs_transformer.transform_documents(docs, 
                                                          tags_to_extract=["span"])
    
    # 웹스크래핑 내용의 3000 글자 기준으로 내용 스플릿, 오버랩 없음.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=3000, 
                                                                    chunk_overlap=0)
    splits = splitter.split_documents(docs_transformed)
    
    # 반환할 결과를 담을 리스트
    extracted_contents = []
    
    # 스플릿을 순회하며 schema 에 따라 내용 추출
    for split in splits:
        # 각 스플릿에 대하여 스키마 기반 내용 추출
        extracted_content = extraction_chain.run(split.page_content)
        extracted_contents.extend(extracted_content)
        
    return extracted_contents

# 스키마 정의
schema = {
    "properties": {
        "뉴스기사_제목": {"type": "string"},
        "뉴스기사_카테고리": {"type": "string"},
        "뉴스기사_키워드": {"type": "string"},
    },
    "required": ["뉴스기사_제목", "뉴스기사_카테고리", "뉴스기사_키워드"],
}

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
os.environ['HF_HOME'] = r'D:\Shin\NLP\models\HuggingFace Models'

# HuggingFace Model ID
model_id = 'beomi/KoRWKV-1.5B'

# HuggingFacePipeline 객체 생성
llm = huggingface_hub.HuggingFaceHub(
    repo_id=model_id, 
    # device=0,               # -1: CPU(default), 0번 부터는 CUDA 디바이스 번호 지정시 GPU 사용하여 추론
    task="text-generation", # 텍스트 생성
    model_kwargs={"temperature": 0.0, 
                #   "max_length": 64
                  },
)

# 문서내용 추출 체인객체 생성
ext_chain = create_extraction_chain(schema=schema, llm=llm)

# 웹스크래핑 URL 정의
urls = ["https://news.naver.com/section/105"]

# 웹스크래핑 및 스키마 기반 내용 추출
extracted_content = scrape_with_playwright(urls, ext_chain)

# 결과 출력
pprint.pprint(extracted_content)