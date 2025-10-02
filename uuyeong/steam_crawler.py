#!/usr/bin/env python
# coding: utf-8


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

def create_output_dir():
    output_dir = "./game_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def crawl_game_info(driver, game_url, game_name):
    try:
        driver.get(game_url)
        time.sleep(3)
        
        game_info = {
            'name': game_name,
            'url': game_url,
            'reviews': '',
            'price': '',
            'description': '',
            'requirements': ''
        }
        
        # 리뷰 정보 수집
        try:
            reviews_elem = driver.find_elements(By.CSS_SELECTOR, '.user_reviews_summary_row')
            reviews_text = []
            for review in reviews_elem[:3]:  # 상위 3개만
                reviews_text.append(review.text)
            game_info['reviews'] = ' '.join(reviews_text)
        except:
            game_info['reviews'] = '리뷰 정보 없음'
        
        # 가격 정보
        try:
            price_elem = driver.find_element(By.CSS_SELECTOR, '.game_purchase_price')
            game_info['price'] = price_elem.text
        except:
            try:
                price_elem = driver.find_element(By.CSS_SELECTOR, '.discount_final_price')
                game_info['price'] = price_elem.text
            except:
                game_info['price'] = '가격 정보 없음'
        
        # 게임 설명
        try:
            desc_elem = driver.find_element(By.CSS_SELECTOR, '.game_description_snippet')
            game_info['description'] = desc_elem.text
        except:
            game_info['description'] = '설명 없음'
        
        # 시스템 요구사양
        try:
            req_elem = driver.find_element(By.CSS_SELECTOR, '.game_area_sys_req')
            game_info['requirements'] = req_elem.text
        except:
            game_info['requirements'] = '시스템 요구사양 정보 없음'
        
        return game_info
    
    except Exception as e:
        print(f"Error crawling {game_name}: {e}")
        return None

def save_game_data(game_info, output_dir):
    """게임 정보를 txt 파일로 저장"""
    filename = f"{game_info['name'].replace(' ', '_').replace(':', '')}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"게임 이름: {game_info['name']}\n\n")
        f.write(f"URL: {game_info['url']}\n\n")
        f.write(f"가격: {game_info['price']}\n\n")
        f.write(f"리뷰 정보:\n{game_info['reviews']}\n\n")
        f.write(f"게임 설명:\n{game_info['description']}\n\n")
        f.write(f"시스템 요구사양:\n{game_info['requirements']}\n")
    
    print(f"{game_info['name']} 데이터 저장 완료")

def main():
    """메인 실행 함수"""
    # 출력 디렉토리 생성
    output_dir = create_output_dir()
    
    games = [
        ("Hollow Knight", "https://store.steampowered.com/app/367520/Hollow_Knight/"),
        ("Stardew Valley", "https://store.steampowered.com/app/413150/Stardew_Valley/"),
        ("Terraria", "https://store.steampowered.com/app/105600/Terraria/"),
        ("Hades", "https://store.steampowered.com/app/1145360/Hades/"),
        ("Portal 2", "https://store.steampowered.com/app/620/Portal_2/"),
        ("The Witcher 3", "https://store.steampowered.com/app/292030/The_Witcher_3_Wild_Hunt/"),
        ("Celeste", "https://store.steampowered.com/app/504230/Celeste/"),
        ("Red Dead Redemption 2", "https://store.steampowered.com/app/1174180/Red_Dead_Redemption_2/"),
    ]
    
    print("Steam 게임 크롤링 시작...")
    
    driver = webdriver.Chrome()
    
    try:
        for game_name, game_url in games:
            print(f"\n크롤링 중: {game_name}")
            game_info = crawl_game_info(driver, game_url, game_name)
            
            if game_info:
                save_game_data(game_info, output_dir)
            
            time.sleep(2) 
        
        print("\n모든 게임 데이터 크롤링 완료!")
        print(f"데이터 저장 위치: {output_dir}/")
    
    finally:
        driver.quit()
        print("\n브라우저 종료")

if __name__ == "__main__":
    main()

