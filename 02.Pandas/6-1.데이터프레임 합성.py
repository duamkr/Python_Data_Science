
import numpy as np
import pandas as pd

# merge / 데이터 프레임 병합, 기본 merge방식은 inner join 방식
df1 = pd.DataFrame({
    '고객번호': [1001, 1002, 1003, 1004, 1005, 1006, 1007],
    '이름': ['둘리', '도우너', '또치', '길동', '희동', '마이콜', '영희']
}, columns=['고객번호', '이름'])
df1

df2 = pd.DataFrame({
    '고객번호': [1001, 1001, 1005, 1006, 1008, 1001],
    '금액': [10000, 20000, 15000, 5000, 100000, 30000]
}, columns=['고객번호', '금액'])
df2

df1
df2

pd.merge(df1,df2)
# 기본 pd.merge를 실행하면 두 프레임간 공통열 '고객번호'로 join 함.
# 이떄 기본적으로 적용되는 join 은 inner join 방식으로 매칭되는 값만 합쳐짐

#  outer join 기법 / pd.merge(df1, df2, how = 'outer'), 매칭되지 않는 부분도 가져옴
pd.merge(df1,df2, how = 'outer')

# left join /  합치는 프레임의 첫번째 데이터값을 기준으로 merge
pd.merge(df1,df2, how = 'left')

# right join / 합치는 프레임의 두번째 데이터값을 기준으로 merge
pd.merge(df1, df2 , how = 'right')

# 만약 테이블에 키값이 같은 데이터가 여러개 존재하면 모든 경우의 수를 조합
df1 = pd.DataFrame({
    '품종': ['setosa', 'setosa', 'virginica', 'virginica'],
    '꽃잎길이': [1.4, 1.3, 1.5, 1.3]},
    columns=['품종', '꽃잎길이'])
df1

df2 = pd.DataFrame({
    '품종': ['setosa', 'virginica', 'virginica', 'versicolor'],
    '꽃잎너비': [0.4, 0.3, 0.5, 0.3]},
    columns=['품종', '꽃잎너비'])
df2

# df1는 2개의 품종에 대한 꽃잎 길이
# df2는 3개의 품종에 대한 꽃잎 너비
pd.merge(df1,df2)

# merge 대상의 데이터 프레임 중 join 되어야 할 열을 지정해줌 on = ' '
# df1, df2 의 고객명이 join 되야 하지만, 데이터라는 열도 있기 떄문에 on으로 join 열을 지정해줌
df1 = pd.DataFrame({
    '고객명': ['춘향', '춘향', '몽룡'],
    '날짜': ['2018-01-01', '2018-01-02', '2018-01-01'],
    '데이터': ['20000', '30000', '100000']})
df1

df2 = pd.DataFrame({
    '고객명': ['춘향', '몽룡'],
    '데이터': ['여자', '남자']})
df2

pd.merge(df1,df2, on='고객명')

# 혹은 join 대상의 두프레임의 열의 명칭이 다를 경우 left_on, right_on으로 지정해줌
df1 = pd.DataFrame({
    '이름': ['영희', '철수', '철수'],
    '성적': [1, 2, 3]})
df1

df2 = pd.DataFrame({
    '성명': ['영희', '영희', '철수'],
    '성적2': [4, 5, 6]})
df2

pd.merge(df1,df2, left_on = '이름', right_on = '성명')

#  열 기준이 아닌 인덱스를 기준열로 사용시  left_index, right_index 인수를 True

df1 = pd.DataFrame({
    '도시': ['서울', '서울', '서울', '부산', '부산'],
    '연도': [2000, 2005, 2010, 2000, 2005],
    '인구': [9853972, 9762546, 9631482, 3655437, 3512547]})
df1

df2 = pd.DataFrame(
    np.arange(12).reshape((6, 2)),
    index=[['부산', '부산', '서울', '서울', '서울', '서울'],
           [2000, 2005, 2000, 2005, 2010, 2015]],
    columns=['데이터1', '데이터2'])
df2

pd.merge(df1, df2, left_on = ['도시', '연도'], right_index = True)

# 마찬가지도 다른 예

df1 = pd.DataFrame(
    [[1., 2.], [3., 4.], [5., 6.]],
    index=['a', 'c', 'e'],
    columns=['서울', '부산'])
df1

df2 = pd.DataFrame(
    [[7., 8.], [9., 10.], [11., 12.], [13, 14]],
    index=['b', 'c', 'd', 'e'],
    columns=['대구', '광주'])
df2

pd.merge(df1, df2, how='outer', left_index=True, right_index=True)



## join 메서드 / merge 명렁어 대신 join 메서드 사용가능

df1.join(df2, how = 'outer')



## concat 명령을 사용한 데이터 연결
# 기준열을 사용하지 않고 단순히 데이터 연결
# 단순히 두 Series나 DataFrame을 연결하기 때문에 인덱스가 중복될 수 있음(붙임)
s1 = pd.Series([0,1], index =['A','B'])
s2 = pd.Series([2,3,4], index =['A','B','C'])

pd.concat([s1, s2])

# 옆으로 연결시 axis = 1로 인수를 설정
df1 = pd.DataFrame(
    np.arange(6).reshape(3, 2),
    index=['a', 'b', 'c'],
    columns=['데이터1', '데이터2'])
df1

df2 = pd.DataFrame(
    5 + np.arange(4).reshape(2, 2),
    index=['a', 'c'],
    columns=['데이터3', '데이터4'])
df2

pd.concat([df1, df2], axis=1)