
# 실제로 웹에서 크롤링한 데이터를 모아서 본인이 직접 FAQ를 생성해보기
# 필수적으로 사용할 기능
# 웹 크롤링으로 원천 데이터 확보하기
# Embedding 및 Chroma DB 사용법 실습
# 자유 주제이며 실습때 구현한 내용을 충분히 활용 및 복습해보기

# 1. 웹 크롤링으로 원천 데이터 확보하기 by using Selenium & BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_dynamic_page_source(url, timeout=10):
    """Selenium을 사용하여 동적 페이지 소스를 가져오는 함수"""
    driver = None
    try:
        # Headless 모드로 실행하여 브라우저 창을 띄우지 않음
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        driver = webdriver.Chrome(options=options)
        driver.get(url)
        
        wait = WebDriverWait(driver, timeout)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'ul[class*="news_list"]')))
        
        print("✅ 동적 콘텐츠 로딩 완료.")
        return driver.page_source

    except TimeoutException:
        print("❌ 페이지 로딩 시간 초과: 뉴스 목록을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"❌ Selenium 오류 발생: {e}")
        return None
    finally:
        if driver:
            driver.quit()
            print("✅ Selenium 드라이버 종료.")

# 실제 크롤링할 URL
url = "https://m.sports.naver.com/kbaseball/news?sectionId=kbo&team=SS&sort=latest&date=20250926&isPhoto=Y"
page_source = get_dynamic_page_source(url)

if page_source is None:
    print("❌ 웹페이지 소스 가져오기 실패")
    exit(1)

soup = BeautifulSoup(page_source, 'html.parser')  
# CSS 선택자 수정 (동적 클래스 이름 대응)
articles = soup.select('.NewsItem_info_area__Dj4oW')

if not articles:
    print("❌ 뉴스 기사를 찾을 수 없습니다. CSS 선택자를 확인해주세요.")
    exit(1)

print(f"✅ {len(articles)}개의 뉴스 기사를 발견했습니다.")

data = []
successful_crawls = 0

for i, article in enumerate(articles, 1):
    try:
        # article의 자식 em 요소 선택 
        title_element = article.select_one('em')
        summary_element = article.select_one('p')  
        
        if title_element and summary_element:
            title = title_element.get_text(strip=True)
            summary = summary_element.get_text(strip=True)
            
            # 데이터 검증
            if title and summary and len(title) > 3 and len(summary) > 10:
                data.append({'title': title, 'summary': summary})
                successful_crawls += 1
            else:
                print(f"⚠️  기사 {i}: 데이터가 불완전함 (제목: {len(title)}자, 요약: {len(summary)}자)")
        else:
            print(f"⚠️  기사 {i}: 제목 또는 요약을 찾을 수 없음")
            
    except Exception as e:
        print(f"⚠️  기사 {i} 처리 중 오류: {e}")

# 데이터 검증 및 저장
if data:
    df = pd.DataFrame(data)
    df.to_csv('sports_news.csv', index=False, encoding='utf-8-sig')  # 한글 인코딩 추가
    print(f"✅ 총 {successful_crawls}개의 기사가 성공적으로 수집되었습니다.")
    print(f"✅ CSV 파일이 생성되었습니다: sports_news.csv")
else:
    print("❌ 수집된 데이터가 없습니다. 크롤링을 종료합니다.")
    exit(1) 



# 2. Embedding 및 Chroma DB 사용법 실습
