"""
Enhanced RAG with Prompt Engineering
"""

import os
from openai import OpenAI
from build_vector_db import get_embedding
import chromadb
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 기존 DB 사용
dbclient = chromadb.PersistentClient(path="./chroma_db")
collection = dbclient.get_collection("rag_collection")


def expand_query(query):
    """Query Expansion"""
    prompt = f"다음 질문을 3가지로 재표현: {query}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    queries = [query]
    for line in response.choices[0].message.content.split('\n'):
        if line.strip():
            cleaned = line.split('.', 1)[-1].strip()
            if cleaned and cleaned not in queries:
                queries.append(cleaned)
    return queries[:4]


def retrieve_documents(query, top_k=5):
    """문서 검색"""
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    if not results["documents"][0]:
        return []

    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results.get("distances", [[0]*len(results["documents"][0])])[0]
    ):
        docs.append({
            "content": doc,
            "metadata": meta,
            "similarity": 1 / (1 + dist)
        })
    return docs


def rerank_documents(query, documents):
    """Re-ranking"""
    if len(documents) <= 1:
        return documents

    docs_text = "\n".join([f"[Doc{i+1}] {d['content'][:150]}"
                           for i, d in enumerate(documents)])
    prompt = f"질문: {query}\n\n{docs_text}\n\n각 문서 점수(1-10): Doc1: "

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        scores = []
        for line in response.choices[0].message.content.split('\n'):
            if ':' in line:
                try:
                    scores.append(float(line.split(':')[1].strip().split()[0]))
                except:
                    scores.append(5.0)

        for doc, score in zip(documents, scores[:len(documents)]):
            doc['score'] = score

        documents.sort(key=lambda x: x.get('score', 0), reverse=True)
    except:
        pass

    return documents


def generate_answer(query, use_expansion=True, use_reranking=True, top_k=3):
    """답변 생성"""

    if use_expansion:
        queries = expand_query(query)
        all_docs = []
        seen = set()
        for q in queries:
            for doc in retrieve_documents(q, top_k=3):
                if doc['content'] not in seen:
                    all_docs.append(doc)
                    seen.add(doc['content'])
    else:
        all_docs = retrieve_documents(query, top_k=top_k*2)

    if use_reranking:
        all_docs = rerank_documents(query, all_docs)

    docs = all_docs[:top_k]

    if not docs:
        return "관련 문서를 찾지 못했습니다."

    context = "\n\n".join([f"[{d['metadata']['filename']}]\n{d['content']}"
                           for d in docs])

    system_prompt = """문서 기반 답변 시스템.
원칙: 1) 문서 내용만 사용 2) 출처 명시 3) 단계적 추론 4) 없는 정보는 "정보 없음"
형식: 핵심답변 -> 근거 -> 추가정보 -> 한계"""

    user_prompt = f"문서:\n{context}\n\n질문: {query}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def main():
    print("Enhanced RAG Chatbot (quit으로 종료)\n")

    while True:
        query = input("\n질문: ").strip()

        if not query:
            continue

        if query.lower() in ['quit', 'exit']:
            break

        print("\n생성 중...")
        answer = generate_answer(query)
        print("\n" + "="*70)
        print(answer)
        print("="*70)


if __name__ == "__main__":
    main()
