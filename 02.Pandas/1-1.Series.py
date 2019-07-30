
import numpy as np
import pandas as pd

# Series ( 값 value + 인덱스 index)   /
s = pd.Series([9904312, 3448737, 2890451, 2466052],
              index =['서울', '부산', '인천','대구'])

# 인덱스를 지정하지 않으면 0부터 시작하는 정수로 인덱스 입력됨
s = pd.Series([9904312, 3448737, 2890451, 2466052])

pd.Series(range(10,14))

# Series 인덱스는 .index 속성으로 접근 가능

s.index

# Series 값은 .values 로 값 접근
s.values


# Series 데이터에 이름을 부여 .name , 혹은 인덱스에 이름을 부여할 수 있음 .index.name
s.name = "인구"
s.index.name = "도시"
s

# Series 연산 / Series의 연산은 값에만 영향을 미침
s / 1000000


# Serires 인덱싱 / 인덱스 라벨 이용 인덱싱 가능

s[1]
s['부산']

s[3]
s["대구"]

# 배열 인덱싱 - 자료의 순서를 바꾸거나 특정한 자료만 선택 가능
s
s[[0,3,1]]     # 0 - 서울 , 3 - 대구 , 1 - 부산 순으로 출력
s[["서울", "대구", "부산"]]

s[(250e4 < s) & (s < 500e4)]      # 특정한 조건 부여 가능

# 슬라이싱 경우 []안에 인덱스값과 문자열에 따라 불러오는 값이 다름
s[1:3] # 이런경우 1~2까지의 부산, 인천 만 불러오지만
s["부산":"대구"]     # 문자열로 슬라이싱 하는경우 해당 문자까지 다불러옴


# 만약 인덱스 명이 영문인 경우 . 을 이용하여 접근 가능
s0 = pd.Series(range(3), index = ["a", "b", "c"])
s0

s0.a
s0.b
s0.c


# Series와 딕셔너리 자료형은 라벨값을 key로 갖는 측면에서 자료형이 같다고 할수 있음
# 따라서 딕셔너리 자료형에서 제공하는 in , items, for loop 등을 통해 키와 값에 접근이 가능

"서울" in s
"대전" in s

for k, v in s.items() :
    print("%s = %d" % (k,v))

# 딕셔너리 개체에서 시리즈를 만들 수 있음

s2 = pd.Series({"서울":9631482, "부산": 3393191, "인천": 2632035, "대전": 1490158})
s2

# 딕셔너리는 순서를 갖지 않음으로 순서를 지정하고 싶을땐 index를 쓰면됨
s2 = pd.Series({"서울": 9631482, "부산": 3393191, "인천": 2632035, "대전": 1490158},
               index=["부산", "서울", "인천", "대전"])
s2

# Series 인덱스 기반 연산
# 서로 인덱스가 맞지 않으면 NaN으로 표시
ds = s - s2
ds
s.values - s2.values

# 인덱스가 맞지 않거나 서로 없는 경우 NaN표시되는 부분 해결 .notnull 메서드
ds.notnull()         # NaN이면 True , NaN이 아니면 False
ds[ds.notnull()]     # NaN은

# s와 s2를 이용한 인구증가율

rs = (s - s2) / s2 * 100
rs = rs[rs.notnull()]
rs


# 데이터의 갱신, 추가, 삭제
rs['부산'] = 1.63
rs
rs['대구'] = 1.41
rs

del rs['서울']
rs
