import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

class RecipeAssistantSystem:
    def __init__(self):
        """요리 비서 시스템 초기화"""
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.dbclient = None
        self.collection = None
    
    def get_embedding(self, text, model="text-embedding-3-large"):
        """텍스트를 임베딩 벡터로 변환"""
        response = self.client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    
    def chunk_text(self, text, chunk_size=400, chunk_overlap=50):
        """텍스트를 청크 단위로 분할"""
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
    
    def init_vector_db(self, db_path="./chroma_db"):
        """벡터 데이터베이스 초기화"""
        self.dbclient = chromadb.PersistentClient(path=db_path)
        self.collection = self.dbclient.create_collection(name="recipe_assistant_collection", get_or_create=True)
        print("요리 비서 벡터 데이터베이스가 초기화되었습니다.")
    
    def build_vector_db(self):
        """크롤링된 데이터로 벡터 DB 구축"""
        if not self.collection:
            self.init_vector_db()
        
        # source_data 폴더의 텍스트 파일들 로드
        docs = []
        for filename in os.listdir('./source_data'):
            if filename.endswith('.txt'):
                file_path = os.path.join('./source_data', filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    docs.append((filename, text))
        
        doc_id = 0
        for filename, text in docs:
            chunks = self.chunk_text(text, chunk_size=400, chunk_overlap=50)
            for idx, chunk in enumerate(chunks):
                doc_id += 1
                embedding = self.get_embedding(chunk)
                
                self.collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{"filename": filename, "chunk_index": idx}],
                    ids=[str(doc_id)]
                )
        
        print("벡터 데이터베이스 구축이 완료되었습니다.")
    
    def retrieve(self, query, top_k=3):
        """쿼리에 대한 유사 문서 검색"""
        query_embedding = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
    
    def generate_answer(self, query, top_k=3):
        """RAG를 활용한 답변 생성"""
        results = self.retrieve(query, top_k)
        found_docs = results["documents"][0]
        found_metadatas = results["metadatas"][0]
        
        # 컨텍스트 구성
        context_texts = []
        for doc_text, meta in zip(found_docs, found_metadatas):
            context_texts.append(f"<<출처: {meta['filename']}>>\n{doc_text}")
        context_str = "\n\n".join(context_texts)
        
        # 요리 비서 시스템 프롬프트
        system_prompt = """
        당신은 사용자의 질문에 따라 레시피를 추천하고 요리법을 알려주는 친절한 '요리 비서'입니다.
        다음 원칙을 반드시 지켜 답변해주세요:

        1. 반드시 제공된 레시피 문서에 기반해서만 답변합니다. 문서에 없는 내용은 추측하지 않습니다.
        2. 답변은 아래 순서와 섹션 제목을 정확히 사용하세요:
           [재료] → [도구] → [만드는 법] → [팁]
        3. 단계는 번호를 붙여 간결하고 명확하게 서술합니다.
        4. 문서에 해당 정보가 없으면 해당 섹션에서 "제공된 정보에는 해당 내용이 없습니다"라고 답합니다.
        5. 답변 마지막 줄에 출처를 표기하세요. 형식: 출처: 'OOO 레시피' (파일명 또는 문서 내 레시피명 활용)
        """
        
        user_prompt = f"""아래는 제공된 단일 레시피 문서의 내용 일부(청크)입니다. 이 내용에서만 근거를 찾으세요:
        {context_str}
        사용자가 문의한 내용: {query}

        위 지침에 따라 [재료], [도구], [만드는 법], [팁] 섹션을 포함해 답변하고, 마지막 줄에 출처를 표기하세요."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        return response.choices[0].message.content
    
    def run_qa_session(self):
        """순두부 찌개 Q&A 세션 실행"""
        print("=== 요리 비서 시스템 ===")
        print("순두부 찌개개에 대한 질문을 입력하세요 (종료: quit)")
        print("예시: '순두부 찌개 만드는 법 알려줘줘'")
        
        while True:
            user_query = input("\n질문: ")
            if user_query.lower() == "quit":
                break
            
            try:
                answer = self.generate_answer(user_query)
                print(f"\n답변: {answer}")
            except Exception as e:
                print(f"오류 발생: {e}")
    
def main():
    """메인 실행 함수"""
    recipe_system = RecipeAssistantSystem()
    
    try:
        print("=== 벡터 데이터베이스 구축 ===")
        recipe_system.build_vector_db()
        
        print("\n=== 요리 비서 Q&A 세션 시작 ===")
        recipe_system.run_qa_session()
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
