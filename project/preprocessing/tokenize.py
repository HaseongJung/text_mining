import os
import time
import re
import random as rd
import pandas as pd
import numpy as np
import collections
from tqdm import tqdm
import pickle

from konlpy.tag import Mecab

import multiprocessing as mp
import parmap


def main():
    df = pd.read_pickle("./data/deduplicated_total.pkl")
    print(df)

    mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')

    tqdm.pandas()   # progress_apply를 사용하기 위한 설정
    num_workers = mp.cpu_count()    # 병렬 처리를 위한 worker 수 설정
    print(f'num_workers: {num_workers}')    

    # 형태소 분석
    # df["Tokenized"] = df["Text"].progress_apply(lambda x: mecab.morphs(x))
    df["Tokenized"] = parmap.map(mecab.morphs, df["Text"], pm_pbar=True, pm_processes=num_workers)  # parmap.map 함수를 사용하여 mecab.morphs 함수를 병렬로 적용  

    # 열 인덱스 지정
    df = df[["Query", "Title", "Text", "News", "Tokenized"]]

    print("Tokenized!: ")
    print(df)
    df.to_pickle("./data/tokenized_total.pkl")

if __name__ == "__main__":
    main()