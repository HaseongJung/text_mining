# import the libraries
import time
import re
import random as rd
import pandas as pd
import numpy as np
import collections
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

user_agent = "Mozilla/5.0 (Linux; Android 9; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.83 Mobile Safari/537.36"
headers = {
    "User-Agent": user_agent
}

# 키워드를 입력하여 URL 생성
query = input("검색어를 입력하세요: ")
start_date = re.sub(r'[^0-9]', '', input("시작 날짜를 yyyy.mm.dd 형식으로 입력하세요: ")) # 시작 날짜(정규식을 이용하여 숫자만 추출)
end_date = re.sub(r'[^0-9]', '', input("종료 날짜를 yyyy.mm.dd 형식으로 입력하세요: "))    # 종료 날짜(정규식을 이용하여 숫자만 추출)

num_articles_per_page = 15  # 페이지당 기사 수
num_pages = int(input("크롤링할 페이지 수를 입력하세요: "))  # 크롤링할 페이지 수

# 데이터를 저장할 리스트 초기화
title_list = []

for page in tqdm(range(num_pages), desc='Pages...'):
    start_index = page * num_articles_per_page + 1  # 페이지의 시작 인덱스 계산

    # URL 생성
    # url = f'https://search.naver.com/search.naver?where=news&sm=tab_jum&query={query}&start={start_index}&pd=3&ds={start_date}&de={end_date}'
    url = 'https://search.naver.com/search.naver?where=news&sm=tab_jum&query=' + \
        '%s' % query + "&start=" + str(start_index) + "&pd=3&ds=" + \
        '%s' % start_date + "&de=" + '%s' % end_date

    print(f'URL: {url}')

    # HTML 문서 가져오기
    response = requests.get(url, headers=headers);  print(f'status code: {response.status_code}')
    html = response.text

    # BeautifulSoup을 이용하여 HTML 파싱
    soup = bs(html, 'html.parser'); 

    # 기사 제목 추출
    titles = soup.find_all('a', class_="news_tit")

    # 추출한 데이터를 리스트에 저장
    for title in titles:
        print(title.text)
        title_list.append(title.text)
        
    print(f'{page}번째 페이지')
    print(f'기사 갯수: {len(title_list)}')
    print(f'{"-"*200}')
    time.sleep(rd.uniform(2.5, 3.5))
        
print(f'총 기사 갯수: {len(title_list)}')
print(title_list)

# 데이터프레임 생성
df = pd.DataFrame(title_list, columns=["뉴스 제목"])

# CSV 파일로 저장
df.to_csv(f'../data/{query}_{start_date}~{end_date}_{num_pages}page.csv', index=False)