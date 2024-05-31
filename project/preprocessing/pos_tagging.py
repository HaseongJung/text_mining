import pandas as pd
from tqdm import tqdm
import pickle

from konlpy.tag import Mecab

# for parallel processing
import multiprocessing as mp
import parmap
import dask.dataframe as dd
from dask.diagnostics import ProgressBar



def pos_tagging(token: list):

    mecab = Mecab(dicpath='C://mecab//mecab-ko-dic')
    return mecab.pos(''.join(token))    # ' '.join(token)으로 리스트를 문자열로 변환하여 입력

def main():
    
    # df = pd.read_pickle('./data/tokenized_total.pkl')
    df = pd.read_feather('./data/processing/posExtracted_total.feather', use_threads=True)
    print(f'df: \n{df.head()}')
    print('-'*200)

    tqdm.pandas()   # progress_apply를 사용하기 위한 설정
    num_workers = mp.cpu_count()    # 병렬 처리를 위한 worker 수 설정
    print(f'num_workers: {num_workers}')    
    print(f'dic before: \n{df}')

    # # "Tokenized" 열의 단어들에 대해 품사 태깅 수행
    df["Title"] = parmap.map(pos_tagging, df["Title"], pm_pbar=True, pm_processes=num_workers)
    print(f'df after: \n{df.head()}')

    # df.drop('Tokenized', axis=1, inplace=True)
    df.to_feather('./data/processing/posExtracted_total.feather')

if __name__ == '__main__':
    mp.freeze_support()
    main()
