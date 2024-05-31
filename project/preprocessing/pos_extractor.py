import pandas as pd
import multiprocessing as mp
import parmap
# import threading


#추출된 품사 필터링
def filtering(pos_list: list):

    desired_pos = ['NNG', 'NNP', 'VV', 'VA', 'MAG', 'NR', 'MM']     # 일반명사, 고유명사, 동사, 형용사, 일반부사, 수사, 관형사
    
    toekns = []
    for pos in pos_list:
        if pos[1] in desired_pos:
            toekns.append(pos[0])

    return toekns


       

def main():

    dataset_path = './data/processing/posExtracted_total.feather'
    df = pd.read_feather(dataset_path, use_threads=True)
    print(f'before: \n{df.head()}')


    df['Title'] = parmap.map(filtering, df['Title'], pm_pbar=True, pm_processes=mp.cpu_count())
    
    # # multi-threading
    # df["tokens"] = pd.Series([[] for _ in range(len(df))])
    # t = threading.Thread(target=filtering, args=(df,))
    # t.start()
    print(f'df after: \n{df.head()}')

    df.to_feather(dataset_path)
    

if __name__ == '__main__':
    mp.freeze_support()
    main()

    # column name 변경
    # df.rename(columns={"tokenized_Title": "Title"}, inplace=True)