#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -r ../requirements.txt')


# In[2]:


import os  # os를 가져와 파일 시스템 접근, 환경 변수 읽을 수 있음
from openai import OpenAI  # OpenAI의 api 사용 가능
import chromadb  # chromadb 라이브러리 쓸 수 있게 해줌
from chromadb.config import (
    Settings,
)  # Settings 클래스는 DB의 구성 옵션을 설정하는데 사용
from dotenv import load_dotenv  # 환경 변수를 로드하기 위함


# 1. 환경 변수 Load해서 api_key 가져오고 OpenAI 클라이언트(객체) 초기화

# In[3]:


import os # os를 가져와 파일 시스템 접근, 환경 변수 읽을 수 있음
from openai import OpenAI # OpenAI의 api 사용 가능
import chromadb # chromadb 라이브러리 쓸 수 있게 해줌
from chromadb.config import Settings # Settings 클래스는 DB의 구성 옵션을 설정하는데 사용
from dotenv import load_dotenv # 환경 변수를 로드하기 위함

# 환경 변수 Load해서 api_key 가져오고 OpenAI 클라이언트(객체) 초기화
load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

print("API Key Loaded:", api_key is not None) 


# 2. DB 초기화 함수 (매 실행 시 DB 폴더를 삭제 후 새로 생성)

# In[4]:


# DB 초기화 함수
def init_db(db_path="./chroma_db"): # 현재 디렉토리 내의 chroma_db폴더 설정
    dbclient = chromadb.PersistentClient(path=db_path) # 지정 경로로 향하는 dbClient 생성
    # rag_collection이라는 데이터 컬렉션(모음집) 만듦
    # get_or_create옵션은 만약 해당 이름 컬렉션이 이미 존재하면 기존 컬렉션 쓰고 아님 만들고
    collection = dbclient.create_collection(name="rag_collection", get_or_create=True)
    return dbclient, collection


# 4. 주어진 text를 임베딩 벡터로 변환하는 함수

# In[5]:


# 주어진 text를 임베딩 벡터로 변환하는 함수 
def get_embedding(text, model="text-embedding-3-large"): #openai에서 제공하는 모델 사용
		# 여기서 client는 앞서 초기화한 OpenAI 클라이언트
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding # 응답 객체의 data 리스트에서 embedding 필드 추출
    return embedding 


# 5. 원천 데이터 청크 단위로 나누고 overlap 사이즈 조절하는 함수

# In[6]:


# 원천 데이터 청크 단위로 나누고 overlap 사이즈 조절하는 함수
def chunk_text(text, chunk_size=400, chunk_overlap=50):
    chunks = [] # 분할된 텍스트 청크들을 저장할 리스트
    start = 0 # 청크를 시작할 위치를 나타내는 인덱스
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end] # 텍스트에서 start부터 end까지 부분 문자열을 추출
        chunks.append(chunk) # 추출한 청크를 리스트에 추가
        start = end - chunk_overlap # overlap 적용

        if start < 0: # 음수가 될 수 있으니 예외 처리
            start = 0

        if start >= len(text): # 종료 시그널
            break

    return chunks # 모든 청크가 저장된 리스트를 반환


# CSV 로더

# In[7]:


def load_rows_from_csv(csv_path: str):
    """
    CSV의 각 행을 (row_id, title, description, combined_text) 형태로 반환
    - 컬럼명: Title, Description (대소문자 그대로)
    """
    df = pd.read_csv(csv_path)
    # 혹시 컬럼명이 다른 경우를 대비한 안전장치
    title_col = "Title" if "Title" in df.columns else df.columns[0]
    desc_col  = "Description" if "Description" in df.columns else df.columns[1]

    rows = []
    for i, row in df.iterrows():
        title = str(row.get(title_col, "") or "")
        desc  = str(row.get(desc_col, "") or "")
        # 임베딩에 넣을 결합 텍스트 (제목 가중치 약간 주고 싶으면 앞에 두 번 배치해도 됨)
        combined = f"[TITLE] {title}\n[SUMMARY] {desc}"
        rows.append((i, title, desc, combined))
    return rows


# 6. 문서로드 -> 청크 나누고 -> 임베딩 생성 후 DB 삽입

# In[8]:


import pandas as pd 

# 문서로드 -> 청크 나누고 -> 임베딩 생성 후 DB 삽입
if __name__ == "__main__":
    # db 초기화
    dbclient, collection = init_db("./chroma_db")

    csv_path = "./naverent_movie_data.csv" # 데이터 가져다 쓸 경로 지정
    rows = load_rows_from_csv(csv_path) # 처리할 문서 데이터 메모리로 불러오기

    doc_id = 0
    
    for row_idx, title, desc, text in rows:
        # 기사 길이에 따라 청크 크기 조절
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50) # chunking
        for idx, chunk in enumerate(chunks):
            doc_id +=1
            embedding = get_embedding(chunk)

        collection.add(
            documents=[chunk], # 실제 청크 text
            embeddings=[embedding], # 생성된 임베딩 벡터
            metadatas=[{
                    "source": "naver_ent_movie_csv",
                    "row_index": int(row_idx),
                    "chunk_index": int(idx),
                    "title": title,
                    "description_present": bool(desc.strip()),
            }], # 파일 이름과 청크 인덱스를 포함하는 메타데이터
            ids=[str(doc_id)] # 각 청크의 Unique한 id 저장
            # 이 고유 id를 통해 db에서 업데이트, 삭제등의 작업 가능 
        )

            
    print("모든 문서 벡터DB에 저장 완료")


# 
