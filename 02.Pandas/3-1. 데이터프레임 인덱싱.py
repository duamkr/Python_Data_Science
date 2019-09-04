import numpy as np
import pandas as pd

# 데이터프레임 고급 인덱싱  - .loc 라벨값 기반의 2차원 인덱싱 / df.loc[행 인덱싱값], df.loc[행 인덱싱값, 열 인덱싱값]
df = pd.DataFrame(np.arange(10, 22).reshape(3, 4),
                  index=["a", "b", "c"],
                  columns=["A", "B", "C", "D"])
df

df.loc["a"]        # 'a' 행을 가져옴
df.loc["b":"c"]    # 'b' 행을 가져옴 ( .loc 안써도 같음)
df["b":"c"]


df.loc[["b", "c"]] # 행을 가져올땐 데이터리스트 인덱싱으로 할땐 ":"만 성립되지만, .loc 메서드 사용시 ","로 구분 가능

# 불리언 시리즈로도 행을 선택할 수 있음

df.A > 15      # "A"열 중 15보다 큰값 True, 작은값 False

# 값이 해당되는 행을 가져올 수 있음
df.loc[df.A > 15]    # "A" 열의 값# 이 15보다 큰 행을 가져오는 식

# 함수 사용 가능 , A열의 값이 12보다 큰 행 선택
def select_rows(df):
    return df.A > 15

select_rows(df)    # 함수에 맞는 조건 , 15보다 큰 행 추출 (True, False)
df.loc[select_rows(df)]   # 행단위 출력

df
df.loc["A"]


# 라벨 슬라이싱 방식을 따라 [1:2]이지만 2를 포함해서 가져옴
df2 = pd.DataFrame(np.arange(10, 26).reshape(4, 4), columns=["A", "B", "C", "D"])
df2

df2.loc[1:2]

# 인덱싱 값을 행과 열 모두 받는 경우 .loc["행","열"]
df
df.loc["a","A"]       # a행의 A열의 값 10
df.loc["b":,"A"]      # b열부터 끝까지 ~ A행의 값

df.loc["a", :]
df.loc[["a","b"],["B","D"]]

# iloc 인덱서 , 순서를 나타내는 * 정수 인덱스만 받음

df.iloc[0,1]      # df.loc["a","A"] 와 같음
df.iloc[:2df.iloc[0, -2:],2]
df.iloc[0, -2:]
df.iloc[2:3, 1:3]

# loc와 마찬가지로 인덱스가 하나만 들어가면 행을 선택함
df.iloc[-1]        # -1이면 마지막, 즉 행의 마지막 행 선택