from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import time
import pandas as pd
from selenium.common.exceptions import NoSuchElementException

class reviews_crawling():
    '''
    
    '''

    def __init__(self):
        self.CGV_URL = "http://www.cgv.co.kr/movies/detail-view/?midx="
        self.CGV_OBJECT_NAME = "cgv_reviews.csv"
        self.CGV_DATA_PATH = f"./data/{self.CGV_OBJECT_NAME}"

        # self.MEGABOX_URL = "https://www.megabox.co.kr/movie-detail/comment?rpstMovieNo=24010200"
        self.MEGABOX_OBJECT_NAME = "megabox_reviews.csv"
        self.MEGABOX_DATA_PATH = f"./data/{self.MEGABOX_OBJECT_NAME}"
    
    def get_movie_reviews_on_cgv(self, url:str, page_num:int=10):
        """ CGV에서 특정 영화의 리뷰를 가져옵니다.
        
        :param url: 
        """
        wd = webdriver.Chrome()
        wd.get(url)
        # 빈 리스트 생성하기
        review_list=[]
        
        for page_no in range(1,page_num+1): # 1페이지에서 page_num까지의 리뷰 추출
            try:
                page_ul = wd.find_element(By.ID, 'paging_point') # 페이지 포인트 코드 추출
                page_a = page_ul.find_element(By.LINK_TEXT, str(page_no))
                page_a.click()
                time.sleep(2) # 페이지 로딩까지의 시간 두기

                reviews = wd.find_elements(By.CLASS_NAME, 'box-comment')
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
    
    def get_movie_reviews_on_megabox(self, url, page_num=10):
        wd = webdriver.Chrome()
        wd.get(url)
        # 빈 리스트 생성하기
        review_list=[]
        
        for page_no in range(1,page_num+1): # 1페이지에서 page_num까지의 리뷰 추출
            try:
                if page_no % 10 != 1:
                    page_nav = wd.find_element(By.CLASS_NAME, 'pagination') # 페이지 포인트 코드 추출
                    page_a = page_nav.find_element(By.LINK_TEXT, str(page_no))
                    page_a.click()
                    time.sleep(2) # 페이지 로딩까지의 시간 두기

                reviews = wd.find_elements(By.CLASS_NAME, 'story-txt')
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