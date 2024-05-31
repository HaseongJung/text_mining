import pandas as pd
from tqdm import tqdm
from collections import Counter
from konlpy.tag import Okt
from tqdm import tqdm
import parmap
import requests
import io

# df = pd.read_feather("./data/posExtracted_total.feather")
# column name 변경

# 빈도 분석
# text = []
# for i in tqdm(range(len(df))):
#     text.extend(df["Tokens"][i])

# print(f'len(text): {len(text)}')

# # Counter 객체 생성
# count = Counter(text)
# print(f'count: {count}')

# # class 'collections.Counter'> 객체를 list로 변환
# count_list = list(count.items())
# count_lis = count_list.sort(key=lambda x: x[1], reverse=True) # 내림차순 정렬
# print(f'count_list: {count_list}')

# # csv 파일로 저장
# count_df = pd.DataFrame(count_list, columns=["word", "count"])
# count_df.to_csv("./data/word_count.csv")


def remove_stopwords(tokens, stopwords):
    return [token for token in tokens if (token not in stopwords) and (len(token) > 1)] # 한 글자 이상인 단어만 추출, 불용어 사전으로 제거


def main():
    df = pd.read_feather("./data/processing/posExtracted_total.feather", use_threads=True)
    print(df.head())

    url = "https://gist.githubusercontent.com/chulgil/d10b18575a73778da4bc83853385465c/raw/a1a451421097fa9a93179cb1f1f0dc392f1f9da9/stopwords.txt"  # 불용어 사전
    response = requests.get(url)    # 불용어 사전 다운로드
    data = response.content.decode("utf-8") # 불용어 사전을 utf-8로 디코딩

    stopwords = data.split("\n")    # 불용어 사전을 줄바꿈을 기준으로 분리
    stopwords = [word for word in stopwords if word]    # 빈 문자열 제거
    
    print(f'stopwords: {stopwords}')
    df["Title"] = parmap.map(remove_stopwords, df["Title"], stopwords, pm_pbar=True)
    df["Text"] = parmap.map(remove_stopwords, df["Text"], stopwords, pm_pbar=True)
    
    print(df.head())

    df.to_feather("./data/processing/stopWordsRemoved_total.feather")


if __name__ == "__main__":
    main()


# df = pd.read_feather("./data/posExtracted_total.feather")



# df = pd.read_feather("./data/posExtracted_total.feather")



# okt = Okt()
# #  불용어 제거
# stopwords = pd.read_csv("./data/stopwords.csv")
# stopwords = list(stopwords["stopwords"])
# # print(f'stopwords: {stopwords}')

# # 불용어 제거
# def remove_stopwords(text):
#     result = []
#     for w in text:
#         if w not in stopwords:
#             result.append(w)
#     return result

# # 불용어 제거
# df["Tokens"] = df["Tokens"].apply(lambda x: remove_stopwords(x))

# # 불용어 제거 후 저장
# df.to_feather("./data/stopWordsRemoved_total.feather")