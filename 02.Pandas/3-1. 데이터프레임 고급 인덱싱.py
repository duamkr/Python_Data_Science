import numpy as np
import pandas as pd

# 데이터프레임 고급 인덱싱  - .loc 라벨값 기반의 2차원 인덱싱 / df.loc[행 인덱싱값], df.loc[행 인덱싱값, 열 인덱싱값]
df = pd.DataFrame(np.arange(10, 22).reshape(3, 4),
                  index=["a", "b", "c"],
                  columns=["A", "B", "C", "D"])
df

df.loc["a"]        # 'a' 열을 가져옴
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



