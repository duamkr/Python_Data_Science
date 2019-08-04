import numpy as np
import pandas as pd
print('hello')
# 데이터 갯수 세기
s = pd.Series(range(10))
s[3] = np.nan    # 0~9 값을 가지는 시리즈 생성, 3은 NaN을 가진 변수 생성

s.count()     # NaN은 카운트에서 제외됨


# 0~5값을 가지는
np.random.seed(2)
df = pd.DataFrame(np.random.randint(5, size=(4,4)), dtype=float)
df.iloc[2,3] = np.nan

df.count()     # 각 행의 값의 갯수 카운트 , NaN은 카운트에서 제외

# 카테고리 값 세기

np.random.seed(1)
s2 = pd.Series(np.random.randint(6, size =100))
s2.tail()

s2.value_counts()
df[0].value_counts()      # 데이터 프레임의 경우 value_counts메서드가 없어 각 열마다 별도로 적용(열 1개 = Series이기 떄문)

# 정렬 / sort_index, sort_values

s2.value_counts().sort_index
s.sort_values()

# 내림차순으로 정렬하려면 ascending = False
s.sort_values(ascending = False)



# 행 / 열 합계  sum(axis)
np.random.seed(1)
df2 = pd.DataFrame(np.random.randint(10, size = (4, 8)))
df2

# 행합계 - axis = 1
df2.sum(axis = 1)     # 행의 합계

df2["RowSUme"] = df2.sum(axis=1)
df2

# 열합계 - 열은 기본 axis 의 인수가 0이기 때문에 axis를 생략 할 수있음
df2.sum()

df2.loc["ColTotal", :] = df2.sum()
df2
df2.iloc[-1]

# apply 변환 - 행이나 열단위 더복잡한 계산시 apply 메서드 사용/ 인수로 행또는 열을 받음
df3 = pd.DataFrame({
    'A': [1, 3, 4, 3, 4],
    'B': [2, 3, 1, 2, 3],
    'C': [1, 5, 2, 4, 4]
})
df3

df3.apply(lambda x : x.max() - x.min())     # 각 열의 최대값 - 최솟값
df3.apply(lambda x : x.max() - x.min(),axis = 1)     # 각 행의 최대값 - 최솟값

df3.apply(pd.value_counts)
df3.apply(pd.value_counts,axis = 1)


# 실수값을 카테고리 값으로 변환 / cut, qcut

ages = [0, 2, 10, 21, 23, 37, 31, 61, 20, 41, 32, 100]

bins = [1, 15, 25, 35, 60, 99]     # 카테고리를 나누는 기준값 설정
labels = ["미성년자", "청년", "중년", "장년", "노년"]      # 카테고리의 기준값인 bins 와 자리수가 같도록 설정
cats = pd.cut(ages, bins, labels=labels)         # age리스트를 bins를 기준 분류, 분류된 labels = labels


cats
type(cats)           # Categorical 클래스 객체
cats.categories      # cats 카테고리 확인
cats.codes     # age의 라벨문자열을 codes속성 정수로 인코딩된 카테고리 값, 이는 미셩년이 0,청년 1, 중년 2, 장년 3, 노년 4이며, age의 리스트 순서대로 정수로 나열됨


# 데이터 프레임으로 추가하기,  ages를 데이터프레임으로 만들고,   age_cat 을 cut 으로 정의해줌
df4 = pd.DataFrame(ages, columns = ["ages"])
df4["age_cat"] = pd.cut(df4.ages, bins, labels=labels )
df4



# qcut  /  경계선을 지정하지 않고 데이터 갯수가 같도록 지정, 1000개의 데이터 -> 4개구간으로 나누면 250개씩 나눠 가짐

data = np.random.randn(1000)
cats = pd.qcut(data, 4, labels = ["Q1", "Q2", "Q3", "Q4"])
cats

