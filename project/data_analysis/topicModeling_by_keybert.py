import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer # 단어 토큰화
from sklearn.metrics.pairwise import cosine_similarity  # 코사인 유사도 계산
from sentence_transformers import SentenceTransformer   # 문장 임베딩을 위한 모델

import multiprocessing
import parmap

# 키워드 추출 함수
def extractKeywords(df, model, i: int, top_n: int):
    doc = ' '.join(word for word in (df['Title'][i].tolist() + df["Text"][i].tolist()))  # 제목과 본문을 합친 문서
    # print(f"doc: {doc}")

    n_gram_range = (2, 3)   # n-gram 범위

    count = CountVectorizer(ngram_range=n_gram_range)   # n-gram 벡터화
    count.fit([doc])    # n-gram 벡터화를 위한 단어 사전 구축

    candidates = count.get_feature_names_out()  # n-gram 후보 추출

    doc_embedding = model.encode([doc]) # 문서 임베딩
    candidate_embeddings = model.encode(candidates) # 후보군 임베딩

    top_n = top_n   # 상위 n개 후보군 선택
    distances = cosine_similarity(doc_embedding, candidate_embeddings) # 코사인 유사도 계산
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]] # 상위 n개 후보군 선택

    return keywords


# 정당 추출 함수
def extract_politicalParty(keywords):

    count_dict = {"더불어민주당": 0, "국민의힘": 0} # 정당별 카운트 딕셔너리
    for keyword in keywords:    # 키워드 리스트에서
        for word in keyword.split():    # 키워드를 공백 기준으로 분리하여
            if "국민의힘" in word:  # "국민의힘"이 포함되어 있으면
                count_dict["국민의힘"] += 1  # "국민의힘" 카운트 증가
                
            if "민주당" in word:  # "더불어민주당"이 포함되어 있으면
                count_dict["더불어민주당"] += 1 # "더불어민주당" 카운트 증가

    # print(f"count_dict: {count_dict}")
    if count_dict["더불어민주당"] == count_dict["국민의힘"]:
        return np.nan   # 두 정당이 동일하게 언급되었을 경우 np.nan 반환
    else:    # 더불어민주당과 국민의힘 중 더 많이 언급된 정당을 반환
        return max(count_dict, key=count_dict.get)  
    
def extract_and_assign(i, df, model):
    # dataset_path = "./data/processing/stopWordsRemoved_total.feather"
    # df = pd.read_feather(dataset_path, use_threads=True) # dataset load
    # model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")  # 문장 임베딩 모델
    
    
    keywords = extractKeywords(df=df, model=model, i=i, top_n=7)  # 키워드 추출, 상위 7개
    part_name = extract_politicalParty(keywords)    # 정당 추출
    
    return part_name

def main():
    dataset_path = "./data/processing/stopWordsRemoved_total.feather"
    df = pd.read_feather(dataset_path, use_threads=True) # dataset load
    print("Dataset loaded")

    model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")  # 문장 임베딩 모델
    # model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")  # 문장 임베딩 모델
    
    # for i in tqdm(range(len(df))):
    #     keywords = extractKeywords(df=df, model=model, i=i, top_n=7)  # 키워드 추출, 상위 7개
    #     part_name = extract_politicalParty(keywords)    # 정당 추출
    #     df.loc[i, "Main_topic"] = part_name
    # doc_list = [df.Title[i].tolist()+df.Text[i].tolist() for i in range(len(df))]
    # model_list = [model for _ in range(len(df))]
    # index_list = [i for i in range(len(df))]

    # 병렬 처리
    df['Main_topic'] = parmap.map(extract_and_assign, range(len(df)), df=df, model=model, pm_pbar=True, pm_processes=multiprocessing.cpu_count())

    print(df.head())
    # df.to_feather("./data/dataAnalysis/added_mainTopic.feather")  # 정당 정보 추가된 데이터셋 저장


if __name__ == "__main__":
    main()
