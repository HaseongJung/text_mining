# import the libraries
import time
import random as rd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import datetime
from dateutil.relativedelta import relativedelta
import warnings

from bs4 import BeautifulSoup as bs
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager

from newspaper import Article

# Driver setting
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
headers = {
    "User-Agent": user_agent
}

def setWebdriver():
    service = ChromeService(executable_path=ChromeDriverManager().install())    # 크롬 드라이버 최신 버전 설정

    options = ChromeOptions()
    options.add_argument('user-agent=' + user_agent)
    options.add_argument('--start-maximized') #브라우저가 최대화된 상태로 실행됩니다.
    # options.add_argument('headless') #headless모드 브라우저가 뜨지 않고 실행됩니다.
    #options.add_argument('--window-size= x, y') #실행되는 브라우저 크기를 지정할 수 있습니다.
    #options.add_argument('--start-fullscreen') #브라우저가 풀스크린 모드(F11)로 실행됩니다.
    #options.add_argument('--blink-settings=imagesEnabled=false') #브라우저에서 이미지 로딩을 하지 않습니다.
    options.add_argument('--mute-audio') #브라우저에 음소거 옵션을 적용합니다.
    options.add_argument('incognito') #시크릿 모드의 브라우저가 실행됩니다.
    options.add_experimental_option('excludeSwitches', ['enable-logging']) #브라우저 로그를 출력하지 않습니다.
    driver = webdriver.Chrome(service=service, options=options)

    return driver

# 원하는 횟수만큼 Scroll down
def scrollDown(driver, num: int): 
    
    for _ in tqdm(range(0, num), desc="Scrolling...", leave=False):
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(rd.uniform(0.1, 0.3))    

    return

# start_date부터 end_date까지 3개월 단위로 나누기
def getPeriods(start_date, end_date):
    
    periods = []
    while (start_date < end_date):
        if (start_date + relativedelta(months=3) > end_date):
            periods.append([start_date.strftime("%Y.%m.%d"), end_date.strftime("%Y.%m.%d")])

        else:
            periods.append([start_date.strftime("%Y.%m.%d"), (start_date + relativedelta(months=3)).strftime("%Y.%m.%d")])
        start_date += relativedelta(months=3)
    # print(periods)

    return periods

# 언론사 세팅
def setCompany():

    company_names = ["YTN", "JTBC", "MBC", "SBS", "국민일보", "매일경제", "조선일보", "중앙일보", "아시아경제", "한국경제", "KBS", "한겨례", "경향신문"]
    company_codes = {
        "YTN": "1052",
        "JTBC": "1437",
        "MBC": "1214",
        "SBS": "1055",
        "국민일보": "1005",
        "매일경제": "1009",
        "조선일보": "1023",
        "중앙일보": "1025",
        "아시아경제": "1277",
        "한국경제": "1015",
        "KBS": "1056",
        "한겨례": "1028",
        "경향신문": "1032",
    }

    return company_names, company_codes

# crawlingArticle
def crawlingArticle(driver, url):

    try:
        article = Article(url, language="ko")
        article.download()
        article.parse()

        date = article.publish_date   
        title = article.title
        text = article.text   
    except:
        date = np.nan
        title = np.nan
        text = np.nan

    return date, title, text


def main():
    
    # 언론사 세팅
    company_names, comapny_codes = setCompany()

    # 기간 세팅 (3개월 단위로 나누기)
    start_date = datetime.datetime.strptime("2020.09.01", "%Y.%m.%d")
    end_date = datetime.datetime.strptime("2023.12.31", "%Y.%m.%d")
    periods = getPeriods(start_date, end_date)

    # 검색어 세팅
    querys = ["국민의힘", "더불어민주당"]
    
    
    driver = setWebdriver()

    # 언론사별로 크롤링
    for company in tqdm(company_names[2:], desc="Companies...", leave=False):
        company_code = comapny_codes[company]

        df = pd.DataFrame(columns=["Query", "Date", "Title", "Text"])

        # 검색어별로 크롤링
        for query in tqdm(querys, desc="Querys...", leave=False):
        
            # 기간별로 크롤링
            for period in tqdm(periods, desc="Periods...", leave=False):
                start_date = period[0]
                end_date = period[1]

                url = f'https://search.naver.com/search.naver?where=news&query={query}&sm=tab_clk.jou&sort=0&photo=0&field=0&pd=3&ds={start_date}&de={end_date}&docid=&related=0&mynews=1&office_type=1&office_section_code=1&news_office_checked={company_code}&nso=so%3Ar%2Cp%3Afrom20210520to20240520&is_sug_officeid=0&office_category=0&service_area=0'

                driver.get(url)
                time.sleep(1)

                scrollDown(driver, 99)  

                news_urls = [news.get_attribute('href') for news in driver.find_elements(By.CLASS_NAME, "news_tit")]

                # query_list = date_list = title_list = text_list = []
                for news in tqdm(news_urls, desc="News...", leave=False):
                    date, title, text = crawlingArticle(driver, news)   # 해당 뉴스 제목, 날짜, 본문 크롤링
                    # query_list.append(query);   date_list.append(date);     title_list.append(title);   text_list.append(text)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        df.loc[len(df)] = [query, date, title, text]        # 데이터프레임에 추가
                    # tmp_df = pd.DataFrame({"Query": [query], "Date": [date], "Title": [title], "Text": [text]})
                    # df_list = [df, tmp_df]
                    # df = pd.concat([df for df in df_list if not df.empty])

        df.to_csv(f'./data/{company}.csv')

    driver.quit()

if __name__ == "__main__":
    main()