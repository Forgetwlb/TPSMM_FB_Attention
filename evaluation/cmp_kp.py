import pandas as pd
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('input', nargs=2)

# df1, df2 = parser.parse_args().input
df1, df2 = 'GT.pkl', 'gen.pkl'

df1 = pd.read_pickle(df1)
df2 = pd.read_pickle(df2)

# 然后，对 df1 DataFrame 执行了排序操作。它使用了 sort_values() 方法，按照列名 file_name 和 frame_number 对 DataFrame 进行升序排序。
df1 = df1.sort_values(by=['file_name', 'frame_number'])
#df2 = df1.sort_values(by=['file_name', 'frame_number'], ascending=False)
df2 = df2.sort_values(by=['file_name', 'frame_number'])

#print (df1.shape, df2.shape)



#assert df1.shape == df2.shape
scores = []

for i in range(df1.shape[0]):
    file_name1 = df1['file_name'].iloc[i].split('.')[0]
    file_name2 = df2['file_name'].iloc[i].split('.')[0]
    # 使用 assert 语句进行断言检查。它们用于确保 df1 和 df2 中相应行的文件名和帧号相匹配
    # assert file_name1 == file_name2
    assert df1['frame_number'].iloc[i] == df2['frame_number'].iloc[i]
    if df2['value'].iloc[i] is not None: 
        scores.append(np.mean(np.abs(df1['value'].iloc[i] - df2['value'].iloc[i]).astype(float)))

print ("Average difference: %s" % np.mean(scores))
