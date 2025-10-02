import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

headers = {"User-Agent": "Mozilla/5.0"}
BASE_URL = "https://www.transfermarkt.com/brentford-fc/rekordspieler/verein/1148/wettbewerb_id/alle/position/alle/aktive/alle/detailposition/alle/page/{}"

def scrape_brentford_records(max_page=26):
    players_data = []

    for page in range(1, max_page+1):
        url = BASE_URL.format(page)
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")

        rows = soup.select("table.items tbody tr")
        if not rows:
            break

        for row in rows:
            name_tag = row.select_one("td.hauptlink a")
            if not name_tag:
                continue

            # "zentriert" 데이터 추출
            z_vals = [td.text.strip() for td in row.select("td.zentriert")]
            if len(z_vals) < 4:
                continue  # 안전장치

            players_data.append({
                "name": name_tag.text.strip(),
                "birthdate": z_vals[2],
                "appearances": z_vals[3],
                "goals": z_vals[4],
                "assists": z_vals[5]
            })

        print(f"✅ Page {page} 완료 (누적 {len(players_data)}명)")

    return players_data


if __name__ == "__main__":
    players = scrape_brentford_records(max_page=26)
    df = pd.DataFrame(players)

    save_path = os.path.join("source_data", "brentford_records.txt")
    df.to_csv(save_path, index=False, sep="|", encoding="utf-8")

    print(f"🎉 크롤링 완료! {save_path} 저장됨")
