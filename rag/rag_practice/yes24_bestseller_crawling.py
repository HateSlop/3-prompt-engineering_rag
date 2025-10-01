from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import pandas as pd
import re

BASE = 'http://www.yes24.com'
LIST_URL = BASE + '/24/Category/BestSeller?CategoryNumber=001'

# 목록 페이지 열기
html = urlopen(LIST_URL)
soup = bs(html, "html.parser")
table = soup.find('ul', {'id' : 'yesBestList'})
assert table, "목록 컨테이너(ul#yesBestList)를 찾지 못했습니다."

# 목록에서 상세 링크 + 태그 추출
book_urls = []
tags_map = {}  

for li in table.find_all('li', recursive=False):
    a = li.find('a', {'class': 'gd_name'})
    if not a: 
        continue
    href = a.get('href') or ''
    link = href if href.startswith('http') else (BASE + href)
    book_urls.append(link)


    tag_box = li.select_one('div.info_tag span.tag')
    tags = []
    if tag_box:
        for t in tag_box.find_all('a'):
            txt = t.get_text(strip=True)
            if not txt:
                continue
            tags.append(txt.lstrip('#'))
    tags_map[link] = tags

print("수집한 상세 링크 수:", len(book_urls))
print("예시 태그:", tags_map.get(book_urls[0], []))


dic = []
for index, book_url in enumerate(book_urls, start=1):
    try:
        html = urlopen(book_url)
        soup = bs(html, "html.parser")

        # 제목
        title = soup.find('h2', {'class':'gd_name'})
        title = title.get_text(strip=True) if title else ""

        # 저자
        author_box = soup.find('span', {'class': 'gd_auth'})
        authors = []
        if author_box:
            for a in author_box.find_all('a'):
                if a and a.get_text(strip=True):
                    authors.append(a.get_text(strip=True))
        author = ", ".join(authors)

        # 출판사
        pub_tag = soup.find('span', {'class': 'gd_pub'})
        pub = pub_tag.get_text(strip=True) if pub_tag else ""

        # 가격
        price_txt = ""
        for sel in [
            ('span', {'class':'nor_price'}),
            ('em',   {'class':'yes_m'}),
            ('em',   {'class':'yes_b'}),
            ('span', {'class':'price'})
        ]:
            tag = soup.find(*sel)
            if tag and tag.get_text(strip=True):
                price_txt = tag.get_text(strip=True)
                break
        price_num = re.sub(r'[^0-9]', '', price_txt)
        price = int(price_num) if price_num else None

        tags = tags_map.get(book_url, [])
        tags_str = ";".join(dict.fromkeys(tags)) if tags else ""

        print(f"[{index}] {title} | {author} | {price} | {tags_str}")

        dic.append({
            '제목': title,
            '저자': author,
            '가격': price,
            '태그': tags_str,
            'Link': book_url
        })

    except Exception as e:
        print(f"[스킵] {book_url} -> {e}")
        continue

df = pd.DataFrame(dic, columns=['제목','저자','가격','태그','Link'])
print(df.head(10))
df.to_csv('yes24_bestseller.csv', index=False, encoding='utf-8-sig')
print("CSV 저장 완료 → yes24_bestseller.csv")