import requests
from bs4 import BeautifulSoup

url = "https://namu.wiki/w/%EC%84%9C%EA%B0%95%EB%8C%80%ED%95%99%EA%B5%90"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

# 나무위키 본문: 여러 개의 <div class="ncKr6Z3w IBHTMn5o">
content_divs = soup.find_all("div", class_="ncKr6Z3w IBHTMn5o")

texts = []
for div in content_divs:
    texts.append(div.get_text(separator="\n", strip=True))

# 전체 텍스트 합치기
full_text = "\n\n".join(texts)

# 파일 저장
with open("namuwiki_sogang.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("크롤링 완료! namuwiki_sogang.txt 파일로 저장했습니다.")



###

import re, json

raw = open("./namuwiki_sogang.txt", encoding="utf-8").read()
# 1) 각주/편집/메뉴 제거
clean = re.sub(r"\[[0-9]+\]", "", raw)       # [1], [12] 등
clean = clean.replace("[편집]", "")
clean = re.sub(r"\[ 펼치기 · 접기 ]", "", clean)
clean = re.sub(r"^\s*분류\s*$", "", clean, flags=re.M)

# 2) 섹션 기준 분할(예: "8. 학교 생활 및 문화")
blocks = re.split(r"(?m)^(?:\d+(?:\.\d+)*\.)\s.*$", clean)
heads  = re.findall(r"(?m)^(?:\d+(?:\.\d+)*\.)\s.*$", clean)

docs = []
for i, body in enumerate(blocks[1:], 1):
    title = heads[i-1].strip()
    body  = re.sub(r"\n{3,}", "\n\n", body).strip()
    if len(body) < 50: 
        continue
    docs.append({
        "id": f"sogang_{i}",
        "title": "서강대학교",
        "section": title,
        "text": body,
        "path": title.split()[:1]  # 필요시 개선
    })

with open("sogang_sections.jsonl", "w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")