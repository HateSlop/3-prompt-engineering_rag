#!/usr/bin/env python
# coding: utf-8

# # 웹에서 크롤링한 데이터를 모아서 본인이 직접 FAQ를 생성해보기

# ### 1단계: Selenium으로 웹페이지 로드
# 
# Selenium을 사용하여 Yanolja 리뷰 페이지를 로드하고, 스크롤을 내려서 더 많은 데이터를 가져옵니다.

# In[1]:


get_ipython().system('pip install selenium')
get_ipython().system('pip install bs4')
get_ipython().system('pip install pandas')
get_ipython().system('pip install openpyxl')


# In[2]:


get_ipython().run_line_magic('pip', 'install selenium')


# In[4]:


from selenium import webdriver
import time

# Selenium 드라이버 설정 (Chrome 사용)
driver = webdriver.Chrome()

# 네이버 엔터 영화 페이지로 이동
url = 'https://m.entertain.naver.com/now?sid=222'
driver.get(url)

# 페이지 로딩을 위해 대기
time.sleep(3)

# 스크롤 설정: 페이지 하단까지 스크롤을 내리기
scroll_count = 10  # 스크롤 횟수 설정
for _ in range(scroll_count):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)  # 스크롤 이후 대기


# ### 2단계: 페이지 소스 가져오기
# 웹페이지의 HTML 소스를 가져와서 BeautifulSoup을 사용해 데이터를 파싱합니다.

# In[5]:


from bs4 import BeautifulSoup

# 웹페이지 소스 가져오기
page_source = driver.page_source

# BeautifulSoup를 사용하여 HTML 파싱
soup = BeautifulSoup(page_source, 'html.parser')


# ### 3단계: 기사 제목 추출

# In[8]:


# Title 추출
################################
titles_class = soup.select('.NewsItem_title__BXkJ6')
titles = []

# 각 Title 텍스트 정리 후 추가
for title in titles_class:
    cleaned_text = title.get_text(strip=True).replace('\r', '').replace('\n', '')
    titles.append(cleaned_text)

titles


# ### 4단계: 기사 요약 추출

# In[15]:


# 요약 추출
################################
descriptions_class = soup.select('[class*="NewsItem_description__"]')
descriptions = []

# 각 요약 텍스트 정리 후 추가
for description in descriptions_class:
    cleaned_text = description.get_text(strip=True).replace('\r', '').replace('\n', '')
    descriptions.append(cleaned_text)

descriptions


# ### 5단계: 데이터 정리 및 DataFrame으로 변환
# 수집된 데이터를 Pandas DataFrame으로 변환하여 후속 분석을 용이하게 만듭니다.

# In[16]:


import pandas as pd

# 결합하여 리스트 생성
data = list(zip(titles, descriptions))

# DataFrame으로 변환
df_reviews = pd.DataFrame(data, columns=['Title', 'Description'])
df_reviews


# ### 6단계: 자주 등장하는 단어 추출
# 리뷰 텍스트에서 자주 등장하는 단어를 추출하고, 불용어를 제거하여 분석합니다.

# In[17]:


from collections import Counter
import re

# 불용어 리스트 (한국어)
korean_stopwords = set(['이', '그', '저', '것', '들', '다', '을', '를', '에', '의', '가', '이', '는', '해', '한', '하', '하고', '에서', '에게', '과', '와', '너무', '잘', '또','좀', '호텔', '아주', '진짜', '정말'])

# 모든 리뷰를 하나의 문자열로 결합
all_reviews_text = " ".join(reviews)

# 단어 추출 (특수문자 제거)
words = re.findall(r"[가-힣]+", all_reviews_text)

# 불용어 제거
filtered_words = [w for w in words if w not in korean_stopwords and len(w)>1]

# 단어 빈도 계산
word_counts = Counter(filtered_words)

# 자주 등장하는 상위 15개 단어 추출
common_words = word_counts.most_common(15)

common_words


# ### 7단계: 분석 결과 요약

# In[18]:


# 분석 결과 요약
summary_df = pd.DataFrame({
    'Common Words': [', '.join([f"{word}({count})" for word, count in common_words])]
})

# 최종 DataFrame 결합
final_df = pd.concat([df_reviews, summary_df], ignore_index=True)
final_df


# ### 8단계: Excel 파일로 저장
# 최종 결과를 Excel 파일로 저장합니다.

# In[19]:


# Excel 파일로 저장
final_df.to_excel('naverent_movie_data.xlsx', index=False)


# ### 9단계: 드라이버 종료
# 크롤링이 끝난 후, Selenium 드라이버를 종료합니다.

# In[20]:


# 드라이버 종료
driver.quit()

