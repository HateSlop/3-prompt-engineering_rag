# # 실제로 웹에서 크롤링한 데이터를 모아서 본인이 직접 FAQ를 생성해보기
# # 필수적으로 사용할 기능
# # 웹 크롤링으로 원천 데이터 확보하기
# # Embe    df.to_csv(CSV_PATH, index=Fals# CSV 파일 존재 확인
# # 자유 주제이며 실습때 구현한 내용을 충분히 활용 및 복습해보기

# # 1. 웹 크롤링으로 원천 데이터 확보하기 by using Selenium & BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException
# from bs4 import BeautifulSoup
# import pandas as pd
# import time

# def get_dynamic_page_source(url, timeout=10):
#     """Selenium을 사용하여 동적 페이지 소스를 가져오는 함수"""
#     driver = None
#     try:
#         # Headless 모드로 실행하여 브라우저 창을 띄우지 않음
#         options = webdriver.ChromeOptions()
#         options.add_argument("--headless")
#         options.add_argument("--no-sandbox")
#         options.add_argument("--disable-dev-shm-usage")
#         options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

#         driver = webdriver.Chrome(options=options)
#         driver.get(url)
        
#         wait = WebDriverWait(driver, timeout)
#         wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'ul[class*="news_list"]')))
        
#         print("✅ 동적 콘텐츠 로딩 완료.")
#         return driver.page_source

#     except TimeoutException:
#         print("❌ 페이지 로딩 시간 초과: 뉴스 목록을 찾을 수 없습니다.")
#         return None
#     except Exception as e:
#         print(f"❌ Selenium 오류 발생: {e}")
#         return None
#     finally:
#         if driver:
#             driver.quit()
#             print("✅ Selenium 드라이버 종료.")

# # 실제 크롤링할 URL
# url = "https://m.sports.naver.com/kbaseball/news?sectionId=kbo&team=SS&sort=latest&date=20250926&isPhoto=Y"
# page_source = get_dynamic_page_source(url)

# if page_source is None:
#     print("❌ 웹페이지 소스 가져오기 실패")
#     exit(1)

# soup = BeautifulSoup(page_source, 'html.parser')  
# # CSS 선택자 수정 (동적 클래스 이름 대응)
# articles = soup.select('.NewsItem_info_area__Dj4oW')

# if not articles:
#     print("❌ 뉴스 기사를 찾을 수 없습니다. CSS 선택자를 확인해주세요.")
#     exit(1)

# print(f"✅ {len(articles)}개의 뉴스 기사를 발견했습니다.")

# data = []
# successful_crawls = 0

# for i, article in enumerate(articles, 1):
#     try:
#         # article의 자식 em 요소 선택 
#         title_element = article.select_one('em')
#         summary_element = article.select_one('p')  
        
#         if title_element and summary_element:
#             title = title_element.get_text(strip=True)
#             summary = summary_element.get_text(strip=True)
            
#             # 데이터 검증
#             if title and summary and len(title) > 3 and len(summary) > 10:
#                 data.append({'title': title, 'summary': summary})
#                 successful_crawls += 1
#             else:
#                 print(f"⚠️  기사 {i}: 데이터가 불완전함 (제목: {len(title)}자, 요약: {len(summary)}자)")
#         else:
#             print(f"⚠️  기사 {i}: 제목 또는 요약을 찾을 수 없음")
            
#     except Exception as e:
#         print(f"⚠️  기사 {i} 처리 중 오류: {e}")

# # 데이터 검증 및 저장
# if data:
#     df = pd.DataFrame(data)
#     df.to_csv('sports_news.csv', index=False, encoding='utf-8-sig')  # 한글 인코딩 추가
#     print(f"✅ 총 {successful_crawls}개의 기사가 성공적으로 수집되었습니다.")
#     print(f"✅ CSV 파일이 생성되었습니다: sports_news.csv")
# else:
#     print("❌ 수집된 데이터가 없습니다. 크롤링을 종료합니다.")
#     exit(1) 


# 2. Embedding 및 Chroma DB 사용법 실습
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os

# 현재 파일의 디렉토리를 기준으로 절대 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'sports_news.csv')
DB_PATH = os.path.join(BASE_DIR, 'chroma_db')

# 환경 변수에서 API 키 로드 (보안 강화)
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))
      
if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    print("💡 .env 파일에 다음과 같이 추가하세요: OPENAI_API_KEY=your_api_key_here")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# CSV 파일 존재 확인
if not os.path.exists(CSV_PATH):
    print("❌ sports_news.csv 파일을 찾을 수 없습니다.")
    exit(1)

try:
    print("📚 CSV 파일을 로딩 중...")
    loader = CSVLoader(file_path=CSV_PATH, encoding='utf-8')
    documents = loader.load()
    
    if not documents:
        print("❌ 로딩된 문서가 없습니다.")
        exit(1)
        
    print(f"✅ {len(documents)}개의 문서를 로딩했습니다.")
    
    # 한국어에 최적화된 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # 한국어에 적합한 사이즈
        chunk_overlap=40,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    
    print("📝 문서를 분할 중...")
    texts = text_splitter.split_documents(documents)
    print(f"✅ {len(texts)}개의 텍스트 청크로 분할했습니다.")
    
    print("🤖 임베딩 생성 중...")
    embeddings = OpenAIEmbeddings()
    
    # Chroma DB에 저장
    print("🗄️ Chroma DB에 저장 중...")
    vectordb = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=DB_PATH  # 절대 경로 사용
    )
    vectordb.persist()
    print("✅ 벡터 데이터베이스가 생성되었습니다.")
    
    # 검색 시스템 설정
    retriever = vectordb.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}
    )
    
    # LLM 및 QA 체인 설정
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True  # 소스 문서도 반환
    )
    print("✅ QA 체인이 준비되었습니다.")

except Exception as e:
    print(f"❌ RAG 시스템 구축 중 오류 발생: {e}")
    exit(1)


def test_qa_system():
    """QA 시스템 테스트 함수"""
    test_queries = [
        "최근 야구 경기 소식 알려줘",
        "삼성 라이온즈 디아즈는 홈런 개수는?",
        "삼성 라이온즈 디아즈의 타점 개수는?"
    ]
    
    print("\n" + "="*50)
    print("🤖 QA 시스템 테스트 시작")
    print("="*50)
    
    for i, query in enumerate(test_queries, 1):
        try:
            print(f"\n📝 질문 {i}: {query}")
            print("-" * 30)
            
            # 질의응답 실행
            result = qa_chain({"query": query})
            
            print(f"🤖 답변: {result['result']}")
            
            # 소스 문서 정보 출력
            if 'source_documents' in result:
                print(f"\n📚 참조 문서 {len(result['source_documents'])}개:")
                for j, doc in enumerate(result['source_documents'], 1):
                    content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"  {j}. {content}")
            
        except Exception as e:
            print(f"❌ 질문 처리 중 오류: {e}")
        
        print("\n" + "-" * 50)

# QA 시스템 테스트 실행
if 'qa_chain' in locals():
    test_qa_system()
else:
    print("❌ QA 체인이 초기화되지 않았습니다.")