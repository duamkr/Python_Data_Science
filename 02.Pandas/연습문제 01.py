
import numpy as np
import pandas as pd

# 1.연습 문제 1

# 1-1.임의로 두 개의 시리즈 객체를 만든다. 모두 문자열 인덱스를 가져야 하며 두 시리즈에 공통적으로 포함되지 않는 라벨이 있어야 한다.
# # 1-2. 위에서 만든 두 시리즈 객체를 이용하여 사칙 연산을 한다.

a = pd.Series([95,80, 65, 85],
              index = ["A","B", "C", "D"])
b = pd.Series([65,100, 88, 98],
              index = ["A","C","D","E"])

ab_1 = a + b
print(ab_1[ab_1.notnull()])

ab_2 = a - b
print(ab_1[ab_2.notnull()])

ab_3 = a * b
print(ab_1[ab_2.notnull()])

ab_4 = a / b
print(ab_1[ab_2.notnull()])


# 연습 문제 2. 다음 조건을 만족하는 임의의 데이터프레임을 하나 만든다.

# 2-1 열의 갯수와 행의 갯수가 각각 5개 이상이어야 한다.
# 2-2 열에는 정수, 문자열, 실수 자료형 데이터가 각각 1개 이상씩 포함되어 있어야 한다.

data1 = {
    "A" : [1,2,3,4,5],
    "B" : ['가','나','다','라','마'],
    "C" : [1.1, 2.2, 3.3, 4.4, 5.5],
    "D" : ['암','행','어','사','출'],
    "E" : [194, 294, 223, 496, 301]
}

columns1 =["A","B","C","D","E"]
index1 = ["행","복","하","세","요"]

ex1 = pd.DataFrame(data1, index = index1, columns=columns1)
ex1



# 연습 문제3. 다음 데이터프레임에서 지정하는 데이터를 뽑아내거나 처리하라
data = {
    "국어": [80, 90, 70, 30],
    "영어": [90, 70, 60, 40],
    "수학": [90, 60, 80, 70],
}
columns = ["국어", "영어", "수학"]
index = ["춘향", "몽룡", "향단", "방자"]
df = pd.DataFrame(data, index=index, columns=columns)
df
# 3-1 모든학생의 수학 점수를 시리즈로 나타낸다.
df['수학']
# 3-2 모든 학생의 국어와 영어 점수를 데이터 프레임으로 나타낸다.
df[['국어','영어']]
# 3-3 모든 학생의 각 과목 평균 점수를 새로운 열로 추가한다.
df["평균"] = ((df['국어'] + df['영어'] + df['수학']) / 3)
df
# 3-4 방자의 영어 점수를 80점으로 수정하고 평균 점수도 다시 계산한다.
df["영어"]["방자":"방자"] = 80
df["평균"] = ((df['국어'] + df['영어'] + df['수학']) / 3)
df
# 3-5 춘향의 점수를 데이터프레임으로 나타낸다.
df1 = df.T
df1[["춘향"]]
# 3-6 향단의 점수를 시리즈로 나타낸다.
df1['향단']



