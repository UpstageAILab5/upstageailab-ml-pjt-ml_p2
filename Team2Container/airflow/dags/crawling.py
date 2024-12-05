from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

import time
import pandas as pd

class crawling():
    '''
    
    '''

    def __init__(self):
        self.geckodriver = "/usr/local/bin/geckodriver"
    
    def get_movie_reviews_on_cgv(self, url:str, review_count:int=10):
        """ CGV에서 특정 영화의 리뷰를 가져옵니다.
        
        :param url: 
        """

        # Firefox options
        firefox_options = Options()
        firefox_options.add_argument("--headless")  # GUI 없이 실행
        firefox_options.add_argument("--no-sandbox")  # 샌드박스 비활성화 (필수)
        firefox_options.add_argument("--disable-dev-shm-usage")  # 메모리 문제 방지
        firefox_options.add_argument("--disable-gpu")  # GPU 비활성화 (선택)

        # GeckoDriver 경로 설정
        service = Service(self.geckodriver)  # GeckoDriver가 위치한 경로

        wd = webdriver.Firefox(service=service, options=firefox_options)
        wd.get(url)
        # 빈 리스트 생성하기
        review_list=[]
        page_num = int(review_count / 6) if review_count>6 else 1
        count = review_count%6 if review_count>6 else review_count
        
        for page_no in range(1,page_num+1): # 1페이지에서 page_num까지의 리뷰 추출
            try:
                page_ul = wd.find_element(By.ID, 'paging_point') # 페이지 포인트 코드 추출
                page_a = page_ul.find_element(By.LINK_TEXT, str(page_no))
                page_a.click()
                time.sleep(2) # 페이지 로딩까지의 시간 두기

                reviews = wd.find_elements(By.CLASS_NAME, 'box-comment')
                if page_no == page_num:
                    review_list += [ reviews[idx].text for idx in range(0, count)]
                else:
                    review_list += [ review.text for review in reviews ]

                if page_no % 10 == 0: # 10이상의 값을 만났을 때
                    next_button = page_ul.find_element(By.XPATH, './/button[contains(@class, "btn-paging next")]')
                    next_button.click()
                    time.sleep(2)
            except NoSuchElementException as e:
                    print("불러올 페이지가 없습니다.")
                    print(e)
                    break
        movie_review_df = pd.DataFrame({"review" : review_list})
        wd.close()

        return movie_review_df
    
    def get_movie_reviews_on_megabox(self, url:str, review_count:int=10):
        """ Megabox에서 특정 영화의 리뷰를 가져옵니다.
        
        :param url: 
        """

        # Firefox options
        firefox_options = Options()
        firefox_options.add_argument("--headless")  # GUI 없이 실행
        firefox_options.add_argument("--no-sandbox")  # 샌드박스 비활성화 (필수)
        firefox_options.add_argument("--disable-dev-shm-usage")  # 메모리 문제 방지
        firefox_options.add_argument("--disable-gpu")  # GPU 비활성화 (선택)

        # GeckoDriver 경로 설정
        service = Service(self.geckodriver)  # GeckoDriver가 위치한 경로

        wd = webdriver.Firefox(service=service, options=firefox_options)
        wd.get(url)
        # 빈 리스트 생성하기
        review_list=[]
        page_num = int(review_count / 6) if review_count>6 else 1
        count = review_count%6 if review_count>6 else review_count
        
        for page_no in range(1,page_num+1): # 1페이지에서 page_num까지의 리뷰 추출
            try:
                if page_no % 10 != 1:
                    page_nav = wd.find_element(By.CLASS_NAME, 'pagination') # 페이지 포인트 코드 추출
                    page_a = page_nav.find_element(By.LINK_TEXT, str(page_no))
                    page_a.click()
                    time.sleep(2) # 페이지 로딩까지의 시간 두기

                reviews = wd.find_elements(By.CLASS_NAME, 'story-txt')
                if page_no == page_num:
                    review_list += [ reviews[idx].text for idx in range(0, count)]
                else:
                    review_list += [ review.text for review in reviews ]

                if page_no % 10 == 0: # 10이상의 값을 만났을 때
                    next_button = page_nav.find_element(By.XPATH, './/a[contains(@class, "control next")]')
                    next_button.click()
                    time.sleep(2)
            except NoSuchElementException as e:
                print("불러올 페이지가 없습니다.")
                print(e)
                break
        movie_review_df = pd.DataFrame({"review" : review_list})
        wd.close()

        return movie_review_df