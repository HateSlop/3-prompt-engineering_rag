import os
import import_ipynb
from openai import OpenAI
from chromadb import Client
import chromadb
from db import get_embedding
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()
dbclient = chromadb.PersistentClient(path="./chroma_db")
collection = dbclient.get_or_create_collection("rag_collection")

def retrieve(query, top_k=3):
    query_embedding = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results

def generate_answer_with_context(query, top_k=3):
    # retrieve 함수로 결과 얻고
    # top_k에 대한 documents와 metadatas 리스트로 추출
    results = retrieve(query, top_k)
    found_docs = results["documents"][0]
    found_metadatas = results["metadatas"][0]

    # context 구성
    context_texts = []

    for doc_text, meta in zip(found_docs, found_metadatas):
        context_texts.append(f"<<filename: {meta['filename']}>>\n{doc_text}")

    context_str = "\n\n".join(context_texts)
    
    # 프롬프트 작성
    system_prompt = """
    당신은 주어진 문서 정보를 바탕으로 사용자 질문에 답변하는
    지능형 어시스턴트입니다. 다음 원칙을 엄격히 지키세요:

    1. 반드시 제공된 문서 내용에 근거해서만 답변을 작성하세요.
    2. 문서에 언급되지 않은 내용이라면, 함부로 추측하거나 만들어내지 마세요. 
    - 예를 들어, 문서에 특정 인물, 사건이 전혀 언급되지 않았다면 
    “관련 문서를 찾지 못했습니다” 또는 “정보가 없습니다”라고 답변하세요.
    3. 사실 관계를 명확히 기술하고, 불확실한 부분은 “정확한 정보를 찾지 못했습니다”라고 말하세요.
    4. 지나치게 장황하지 않게, 간결하고 알기 쉽게 설명하세요.
    5. 사용자가 질문을 한국어로 한다면, 한국어로 답변하고, 
    다른 언어로 질문한다면 해당 언어로 답변하도록 노력하세요.
    6. 문서 출처나 연도가 중요하다면, 가능한 정확하게 전달하세요.

    당신은 전문적인 지식을 갖춘 듯 정확하고, 동시에 친절하고 이해하기 쉬운 어투를 구사합니다. 
    """

    user_prompt = f"""아래는 검색된 문서들의 내용입니다:
    {context_str}
    질문: {query}"""

    # ChatGPT 호출
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content
    return answer

if __name__ == "__main__":
    # FAQ 질문 리스트
    faq_questions = [
        "현재 1위 곡은 뭐야?",
        "BLACKPINK 노래 순위 알려줘",
        "임영웅은 몇 위야?",
        "로제 노래도 차트에 있어?"
    ]

    faq_results = []

    for q in faq_questions:
        answer = generate_answer_with_context(q, top_k=3)
        faq_results.append({
            "Question": q,
            "Answer": answer
        })
        # 바로 출력
        print("질문:", q)
        print("답변:", answer)
        print("----------")

import pandas as pd

df_faq = pd.DataFrame(faq_results)
df_faq.to_csv("faq_results.csv", index=False, encoding="utf-8-sig")
print("CSV 저장 완료: faq_results.csv")
