import pandas as pd
from tqdm import tqdm
from collections import Counter
from tqdm import tqdm

df = pd.read_feather("./data/processing/stopWordsRemoved_total.feather", use_threads=True)

text = []
for i in tqdm(range(len(df))):
    text.extend(df["Text"][i])

print(f'len(text): {len(text)}')

# Counter 객체 생성
count = Counter(text)
print(f'count: {count}')

# class 'collections.Counter'> 객체를 list로 변환
count_list = list(count.items())
count_lis = count_list.sort(key=lambda x: x[1], reverse=True) # 내림차순 정렬
print(f'count_list: {count_list}')

# csv 파일로 저장
count_df = pd.DataFrame(count_list, columns=["word", "count"])
count_df.to_csv("./data/word_count.csv")