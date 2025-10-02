#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import import_ipynb
from openai import OpenAI
from as_build_vector_db import get_embedding
from chromadb import Client
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()
dbclient = chromadb.PersistentClient(path="./chroma_db")
collection = dbclient.get_or_create_collection("rag_collection")


#  query를 임베딩해 chroma에서 가장 유사도가 높은 top-k개의 문서 가져오는 함수

# In[3]:


# query를 임베딩해 chroma에서 가장 유사도가 높은 top-k개의 문서 가져오는 함수 
def retrieve(query, top_k=3):
    query_embedding = get_embedding(query) # qeury에 대한 임베딩 생성
    # collection.query 함수로 저장된 문서 임베딩들 중에서
    # query임베딩과 가장 유사한 항목들 검색 
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    ) 
    # 이때 results에는 해당 query 임베딩에 대한 텍스트, 메타데이터, id등이 전부 포함됨 
    return results


# 1) query에 대해 벡터 DB에서 top_k개 문서 retrieval
# 2) 그 문서들을 context로 묶어 GPT에 prompt
# 3) 최종 답변 반환
# 하는 함수

# In[4]:


"""
흐름 
1) query에 대해 벡터 DB에서 top_k개 문서 retrieval
2) 그 문서들을 context로 묶어 GPT에 prompt
3) 최종 답변 반환 
하는 함수 
"""
def generate_answer_with_context(query, top_k=3):

    results = retrieve(query, top_k) # retrieve 함수로 결과 얻기
    # top_k에 대한 documents와 metadatas 리스트로 추출
    found_docs = results["documents"][0] 
    found_metadatas = results["metadatas"][0]

    # context 구성 (검색된 문서들을 하나의 문맥으로 결합)
    context_texts = []
    # zip을 이용해 두 리스트의 같은 인덱스에 있는 값들을 한 쌍으로 묶음
    for doc_text, meta in zip(found_docs, found_metadatas):
        # 안전 접근 (키가 없을 수 있으므로 get 사용)
        title = meta.get("title") or "(제목 없음)"
        row_index = meta.get("row_index", "NA")
        chunk_index = meta.get("chunk_index", "NA")
        source = meta.get("source", "naver_ent_movie_csv")
        context_texts.append(
            f"<<source:{source} row:{row_index} chunk:{chunk_index} title:{title}>>\n{doc_text}"
        )
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

    user_prompt =f"""아래는 검색된 문서들의 내용입니다:
    {context_str}
    질문: {query}"""
    
		# context_str과 query를 같이 프롬프트에 넣어버림
    
    # ChatGPT 호출 
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{"role":"system", "content": system_prompt},
        {"role":"user", "content": user_prompt}]
    )

    answer = response.choices[0].message.content
    return answer


#  RAG 없이 응답하는 함수 

# In[5]:


# def generate_answer_without_context(query):
#      api_key = os.getenv("OPENAI_API_KEY")
#      client = OpenAI(api_key=api_key)

#      response = client.chat.completions.create(
#          model = "gpt-4o-mini",
#          messages=[{"role":"system", "content":"you are helpful assistant"},
#                    {"role":"user", "content": query}]
#      )

#      answer = response.choices[0].message.content 
#      return answer


# In[6]:


if __name__ == "__main__":
    while True:
        user_query = input("질문을 입력하세요(종료: quit): ")
        if user_query.lower() == "quit":
            break
        answer = generate_answer_with_context(user_query, top_k=3)
        # answer = generate_answer_without_context(user_query)
        print("===답변===")
        print(answer)
        print("==========\n")


# In[7]:


from tkinter import *
import tkinter.ttk as ttk
def reset_status():
    label_status.config(text="", foreground="black")

def process_query():
    query = text_input.get("1.0", END).strip()
    print(f"User Query: {query}")
    if query:
        label_status.config(text="질문 처리중...", foreground="blue")
        answer = generate_answer_with_context(query)
        print(f"Answer: {answer}")
        label_status.config(text="처리 완료 ✅", foreground="green")
        root.after(2000, reset_status)
        text_input.delete("1.0", "end-1c")          # 입력창 비우기
        text_output.config(state="normal")
        text_output.delete("1.0", END)
        text_output.insert(END, answer)
        text_output.config(state="disabled")   # 출력창 편집 불가
    else:
        label_status.config(text="질문을 입력해주세요.", foreground="red")

root = Tk()
root.title('RAG 챗봇')
root.geometry('500x700')
root.resizable(False, False)
# 전체 배경 연보라
root.configure(bg="lavender")
# ====== 상단 이미지 영역 ======
img = PhotoImage(file="source/hateslop.png")
label_img = Label(root, image=img, bg="white")
label_img.pack(pady=10)

# ====== 입력 영역 ======
frame_input = Frame(root, padx=10, pady=10)
frame_input.pack(fill="x")

label_input = ttk.Label(frame_input, text="질문 입력", font=("맑은 고딕", 12, "bold"))
label_input.pack(anchor="w")

text_input = Text(frame_input, height=6, font=("맑은 고딕", 11))
text_input.pack(pady=5)

btn = ttk.Button(frame_input, text="전송", command=process_query)
btn.pack(pady=5)

label_status = ttk.Label(frame_input, text="", font=("맑은 고딕", 10))
label_status.pack(anchor="w", pady=5)

separator = ttk.Separator(root, orient="horizontal")
separator.pack(fill="x", padx=10, pady=10)
# ====== 출력 영역 ======
frame_output = Frame(root, padx=10, pady=10)
frame_output.pack(fill="both", expand=True)

label_output = ttk.Label(frame_output, text="답변", font=("맑은 고딕", 12, "bold"))
label_output.pack(anchor="w")

text_output = Text(frame_output, wrap="word", font=("맑은 고딕", 11), state="disabled", height=20, bg="#f9f9f9")
text_output.pack(side="left", fill="both", expand=True)

scrollbar = ttk.Scrollbar(frame_output, command=text_output.yview)
scrollbar.pack(side="right", fill="y")
text_output.config(yscrollcommand=scrollbar.set)

root.mainloop()

