import os
import time
from selenium import webdriver
from bs4 import BeautifulSoup


class RecipeCrawler:
    def __init__(self):
        """레시피 크롤러 초기화"""
        self.driver = None

    def setup_driver(self):
        """Selenium 드라이버 설정"""
        self.driver = webdriver.Chrome()
        print("Chrome 드라이버가 설정되었습니다.")

    def crawl_single_recipe(self, url):
        """레시피 페이지에서 정보 수집"""
        if not self.driver:
            self.setup_driver()

        print(f"레시피 크롤링 시작: {url}")
        self.driver.get(url)
        time.sleep(2)

        recipe_soup = BeautifulSoup(self.driver.page_source, 'html.parser')

        # 요리명
        title = ""
        title_selectors = [
            '.view2_summary h3',
            '.recipe_title',
            'h1',
            '.view2_summary_info h3'
        ]
        for selector in title_selectors:
            title_elem = recipe_soup.select_one(selector)
            if title_elem and title_elem.get_text(strip=True):
                title = title_elem.get_text(strip=True)
                break

        # 재료
        ingredients = []
        ingredient_selectors = [
            '#divConfirmedMaterialArea ul li',
            '.ready_ingre3 ul li'
        ]
        for selector in ingredient_selectors:
            ingredient_elems = recipe_soup.select(selector)
            if ingredient_elems:
                for elem in ingredient_elems:
                    text = elem.get_text(strip=True)
                    if text:
                        ingredients.append(text)
                break

        # 조리도구
        tools = []
        # 우선 "조리도구" 섹션 탐색 시도
        ready_sections = recipe_soup.select('.ready_ingre3')
        for section in ready_sections:
            title_elem = section.select_one('.ready_ingre3_tt')
            if title_elem and '도구' in title_elem.get_text():
                tool_items = section.select('ul li')
                for t in tool_items:
                    t_text = t.get_text(strip=True)
                    if t_text:
                        tools.append(t_text)
                break

        # 조리순서
        instructions = []
        step_wrappers = recipe_soup.select('.view_step_cont')
        if step_wrappers:
            for wrap in step_wrappers:
                body = wrap.select_one('.media-body') or wrap
                step_text = body.get_text(strip=True)
                if step_text:
                    instructions.append(step_text)
        else:
            # fallback
            for elem in recipe_soup.select('.recipe_step_count, .view_step'): 
                txt = elem.get_text(strip=True)
                if txt:
                    instructions.append(txt)

        # Tip
        tips = []
        tip_selectors = [
            '.view_step_tip',
            '.view2_box .tip',
            '.tip'
        ]
        for selector in tip_selectors:
            tip_elems = recipe_soup.select(selector)
            for te in tip_elems:
                t_text = te.get_text(strip=True)
                if t_text:
                    tips.append(t_text)
            if tips:
                break

        data = {
            'title': title,
            'ingredients': ingredients,
            'tools': tools,
            'instructions': instructions,
            'tips': tips,
            'url': url
        }

        return data

    def save_single_recipe(self, item, filename):
        """ 레시피를 지정 포맷으로 저장"""
        os.makedirs('./source_data', exist_ok=True)

        with open(f'./source_data/{filename}', 'w', encoding='utf-8') as f:
            f.write(f"레시피명: {item.get('title','')}\n")
            f.write(f"URL: {item.get('url','')}\n\n")

            if item.get('ingredients'):
                f.write("재료:\n")
                for ing in item['ingredients']:
                    f.write(f"- {ing}\n")
                f.write("\n")

            if item.get('tools'):
                f.write("조리도구:\n")
                for tool in item['tools']:
                    f.write(f"- {tool}\n")
                f.write("\n")

            if item.get('instructions'):
                f.write("조리순서:\n")
                for i, step in enumerate(item['instructions'], 1):
                    f.write(f"{i}. {step}\n")
                f.write("\n")

            if item.get('tips'):
                f.write("TIP:\n")
                for tip in item['tips']:
                    f.write(f"- {tip}\n")
                f.write("\n")

        print(f"레시피 데이터가 {filename}에 저장되었습니다.")

    def close_driver(self):
        if self.driver:
            self.driver.quit()
            print("드라이버가 종료되었습니다.")


def main():
    crawler = RecipeCrawler()
    try:
        print("=== 순두부 찌개 레시피 크롤링 시작 ===")
        recipe_url = "https://www.10000recipe.com/recipe/6912220"
        data = crawler.crawl_single_recipe(recipe_url)
        crawler.save_single_recipe(data, "recipe_6912220.txt")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        crawler.close_driver()


if __name__ == "__main__":
    main()


