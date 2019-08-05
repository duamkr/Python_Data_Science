import pandas as pd
import numpy as np


data = {
    "도시": ["서울", "서울", "서울", "부산", "부산", "부산", "인천", "인천"],
    "연도": ["2015", "2010", "2005", "2015", "2010", "2005", "2015", "2010"],
    "인구": [9904312, 9631482, 9762546, 3448737, 3393191, 3512547, 2890451, 263203],
    "지역": ["수도권", "수도권", "수도권", "경상권", "경상권", "경상권", "수도권", "수도권"]
}
columns = ["도시", "연도", "인구", "지역"]
df1 = pd.DataFrame(data, columns=columns)
df1


# 피봇테이블 생성, 도시/연도/인구 데이터 조합
df1.pivot('도시','연도','인구')

# set_index, unstack을 이용해서도 만들 수 있음(참고)
df1.set_index(['도시','연도'])[['인구']].unstack()



# 그룹 연산 / groupby메서드

np.random.seed(0)
df2 = pd.DataFrame({
    'key1': ['A', 'A', 'B', 'B', 'A'],
    'key2': ['one', 'two', 'one', 'two', 'one'],
    'data1': [1, 2, 3, 4, 5],
    'data2': [10, 20, 30, 40, 50]
})
df2

# .groupby 로  key1 의 A, B의 그룹을 지정해 줌
groups = df2.groupby(df2.key1)
groups
# groupby로 지정 한 group 은 클래스 객채이며 이는 각 그룹 데이터의 인덱스를 저장한 groups의 속성을 가짐
groups.groups

# 그룹의 합계
groups.sum()
# 특정 열에 대한 값의 연산을 할 수 있음 / df2데이터의 data1 열의 값을 key1을 그룹해서 연산
df2.data1.groupby(df2.key1).sum()

# Group by 클래스 객체에서 data1만 선택하여 분석
df2.groupby(df2.key1)["data1"].sum()

# 위에선 특정열만 선택했지만, 다 연산하되 특정열만 출력이 가능
df2.groupby(df2.key1).sum()["data1"]

# key 값을 2개로 지정 가능
df2.data1.groupby([df2.key1,df2.key2]).sum()

# unstack()으로 피봇테이블 형태로 만듬
df2.data1.groupby([df2["key1"], df2["key2"]]).sum().unstack("key2")

# 인구데이터 지역별 합계 (변수의 첫 지정시 특정 열만 지정 가능)
df1["인구"].groupby([df1["지역"], df1["연도"]]).sum().unstack("연도")
df1.인구.groupby([df1["지역"], df1["연도"]]).sum().unstack("연도")



# iris 데이터 활용

import seaborn as sns
iris = sns.load_dataset("iris")
iris

def peak_to_peak_ratio(x):
    return x.max() / x.min()

iris.groupby(iris.species).agg(peak_to_peak_ratio)

# describe 메서드 사용시 다양한 기술통계를 한번에 구함 /  하나의 데이터 프레임이 생성
iris.groupby(iris.species).describe().T

# apply 메서드 사용시 describe는 고정된 값을 구하지만, apply는 데이터 프레임을 생성(원하는 옵션 - 함수)
def top3_petal_length(df):
    return df.sort_values(by="petal_length", ascending=False)[:3]     # 붓꽃 종별 가장 긴 꽃잎 길이(petal length)

iris.groupby(iris.species).apply(top3_petal_length)

