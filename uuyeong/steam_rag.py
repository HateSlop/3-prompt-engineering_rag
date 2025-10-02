#!/usr/bin/env python
# coding: utf-8


import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import shutil

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path, override=True)
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    api_key = api_key.strip().strip('"').strip("'")
    print(f"✓ API 키 로드됨 (길이: {len(api_key)}, 끝: ...{api_key[-4:]})")
else:
    print("API 키를 찾을 수 없습니다!")
client = OpenAI(api_key=api_key)


def init_db(db_path="./chroma_db_steam"):
    """DB 초기화 함수"""
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    dbclient = chromadb.PersistentClient(path=db_path)
    collection = dbclient.create_collection(name="steam_games", get_or_create=True)
    return dbclient, collection


def load_text_files(folder_path):
    docs = []
    if not os.path.exists(folder_path):
        print(f"폴더가 존재하지 않습니다: {folder_path}")
        return docs
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append((filename, text))
    
    return docs


def get_embedding(text, model="text-embedding-3-large"):
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return embedding


def chunk_text(text, chunk_size=500, chunk_overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
        
        if start < 0:
            start = 0
        
        if start >= len(text):
            break
    
    return chunks


def build_vector_db(folder_path="./game_data"):
    """벡터 DB 구축"""
    print("\nVector DB 구축 시작...")
    
    dbclient, collection = init_db("./chroma_db_steam")
    
    docs = load_text_files(folder_path)
    
    if not docs:
        print("로드할 문서가 없습니다!")
        return None, None
    
    print(f"총 {len(docs)}개의 게임 데이터 파일 로드됨")
    
    doc_id = 0
    for filename, text in docs:
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=100)
        print(f"  - {filename}: {len(chunks)}개 청크")
        
        for idx, chunk in enumerate(chunks):
            doc_id += 1
            embedding = get_embedding(chunk)
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"filename": filename, "chunk_index": idx}],
                ids=[str(doc_id)]
            )
    
    print(f"총 {doc_id}개 청크가 Vector DB에 저장 완료\n")
    return dbclient, collection


def retrieve(collection, query, top_k=5):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results


def generate_answer(collection, query, top_k=5):
    results = retrieve(collection, query, top_k)
    found_docs = results["documents"][0]
    found_metadatas = results["metadatas"][0]
    
    context_texts = []
    for doc_text, meta in zip(found_docs, found_metadatas):
        context_texts.append(f"[{meta['filename']}]\n{doc_text}")
    context_str = "\n\n---\n\n".join(context_texts)
    
    system_prompt = """
    당신은 Steam 게임 정보 전문가입니다. 
    주어진 게임 데이터를 바탕으로 정확하고 상세하게 답변하세요.
    
    규칙:
    1. 반드시 제공된 문서 내용만을 기반으로 답변하세요.
    2. 문서에 없는 내용은 추측하지 마세요.
    3. 가격, 평점, 리뷰 등 구체적인 정보를 포함하세요.
    4. 한국어로 친절하고 명확하게 답변하세요.
    """
    
    user_prompt = f"""다음은 Steam 게임 데이터입니다:

{context_str}

질문: {query}
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    
    answer = response.choices[0].message.content
    return answer


def main():
    print("=" * 60)
    print("Steam 게임 RAG 챗봇")
    print("=" * 60)
    
    # Vector DB 구축
    dbclient, collection = build_vector_db("./game_data")
    
    if collection is None:
        print("Vector DB 구축 실패. 프로그램을 종료합니다.")
        return
    
    # 4가지 질문
    questions = [
        "가장 평가가 좋은 게임은 무엇인가?",
        "Hollow Knight라는 게임의 리뷰를 분석하여라",
        "인기 게임 중 가장 비싼 게임은 무엇인가?",
        "스타듀밸리라는 게임을 하기 위해 요구하는 사양은?"
    ]
        
    for i, question in enumerate(questions, 1):
        print("=" * 60)
        print(f"[질문 {i}] {question}")
        print("=" * 60)
        
        answer = generate_answer(collection, question, top_k=5)
        print(f"\n[답변]\n{answer}\n")
    
    print("=" * 60)
    print("✨ 모든 질문에 대한 답변 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

