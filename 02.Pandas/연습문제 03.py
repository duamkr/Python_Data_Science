import numpy as np
import pandas as pd

# 연습 문제
# 1. 모든 행과 열에 라벨을 가지는 5 x 5 이상의 크기를 가지는 데이터프레임을 만든다.
df = pd.DataFrame(np.arange(10, 35).reshape(5, 5),
                  index=["a", "b", "c", "d", "e"],
                  columns=["A", "B", "C", "D","E"])
df

# 2. 10가지 이상의 방법으로 특정한 행과 열을 선택한다.
df[:"a"]
df["A"]
df["a":"e"]
df.loc["a","A"]
df.loc["b":,"A"]
df.loc["a", :]
df.loc[["a","b"],["B","D"]]
df.iloc[0,1]      # df.loc["a","A"] 와 같음
df.iloc[:2]
df.iloc[0, -2:]
df.iloc[2:3, 1:3]