import os
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from sklearn.feature_extraction.text import CountVectorizer  # 단어 토큰화
from sklearn.metrics.pairwise import cosine_similarity  # 코사인 유사도 계산
import torch
from sentence_transformers import SentenceTransformer  # 문장 임베딩을 위한 모델

import multiprocessing as mp
from functools import partial
# from accelerate import PartialState

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
# num_cpu = mp.cpu_count()



# 키워드 추출 함수
def extractKeywords(row, top_n: int, model):
    doc = row

    n_gram_range = (2, 3)  # n-gram 범위

    count = CountVectorizer(ngram_range=n_gram_range)  # n-gram 벡터화
    count.fit([doc])  # n-gram 벡터화를 위한 단어 사전 구축

    candidates = count.get_feature_names_out()  # n-gram 후보 추출

    doc_embedding = model.encode([doc])  # 문서 임베딩
    candidate_embeddings = model.encode(candidates)  # 후보군 임베딩
    top_n = top_n  # 상위 n개 후보군 선택
    distances = cosine_similarity(
        doc_embedding, candidate_embeddings
    )  # 코사인 유사도 계산
    keywords = [
        candidates[index] for index in distances.argsort()[0][-top_n:]
    ]  # 상위 n개 후보군 선택

    return keywords


# 대표 키워드에서 정당 추출 함수
def extract_politicalParty(keywords, threshold: int):
    count_dict = {"더불어민주당": 0, "국민의힘": 0}  # 정당별 카운트 딕셔너리
    for keyword in keywords:  # 키워드 리스트에서
        for word in keyword.split():  # 키워드를 공백 기준으로 분리하여
            if "국민의힘" in word:  # "국민의힘"이 포함되어 있으면
                count_dict["국민의힘"] += 1  # "국민의힘" 카운트 증가

            if "민주당" in word:  # "더불어민주당"이 포함되어 있으면
                count_dict["더불어민주당"] += 1  # "더불어민주당" 카운트 증가

    # print(f"count_dict: {count_dict}")
    if (max(count_dict["더불어민주당"], count_dict["국민의힘"]) - min(count_dict["더불어민주당"], count_dict["국민의힘"]) >= 2):  # 두 정당의 차이가 2 이상일 경우
        return max(count_dict, key=count_dict.get)
    else:
        return np.nan


# 주제 추출 및 정당 할당 함수
def extract_and_assign(row, model): # 
    threshold = 3

    keywords = extractKeywords(row, top_n=15, model=model)  # 키워드 추출, 상위 top_n개
    part_name = extract_politicalParty(keywords, threshold=threshold)  # 정당 추출
    
    return part_name


# 함수 적용 함수
def apply_function(data, func):

    return data.progress_apply(func)


# 병렬 처리 함수
def parallel_apply(data, main_func, func, model):   
    num_cores = 10  # CPU 코어 개수

    # partial 함수를 이용하여 함수에 인자 고정
    func = partial(func, model=model)
    pool_func = partial(main_func, func=func) # partial 함수를 이용하여 함수에 인자 고정

    # 병렬 처리
    with mp.Pool(num_cores) as pool:
        df_split = np.array_split(data, num_cores)
        
        #  
        df = pd.concat(pool.map(func=pool_func, iterable=df_split))
        
        # use pool.apply_async
        # multiple_results = [pool.apply_async(func=pool_func, args=(df_split[i], )) for i in range(num_cores)]
        # df = pd.concat([res.get() for res in multiple_results])

    return df


def main():
    # Load dataset
    dataset_path = "./data/processing/stopWordsRemoved_total.feather"
    df = pd.read_feather(dataset_path, use_threads=True)  # dataset load
    print("Dataset loaded!")

    # Load model
    model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")  # 문장 임베딩 모델
    print("Model loaded!")

    # Combine title and text
    df["Main_topic"] = pd.Series(map(lambda x: ' '.join(x), df.Title)) + pd.Series(map(lambda x: ' '.join(x), df.Text))

    # Extract and assign main topic
    df["Main_topic"] = parallel_apply(df.Main_topic, apply_function, extract_and_assign, model=model)  # 주제 추출 및 정당 할당

    # Save dataset
    df.to_feather("./data/dataAnalysis/added_mainTopic2.feather")  # 정당 정보 추가된 데이터셋 저장


if __name__ == "__main__":
    mp.freeze_support()     
    main()
