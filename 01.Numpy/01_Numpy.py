#!/usr/bin/env python
# coding: utf-8

# # Numpy(19.07.29)
# 
# ## 1. Numpy 개요
# #### 1.1 파이썬 과학 패키지
# 
# 

# ## 2. ndarray(Numpy Dimensional Array)

# In[2]:


import numpy as np


# #### 2.2 Array 생성

# In[279]:


test_array = np.array([1,4,5,8], float)
print(test_array)        # 실수 형태의 test_array 생성 [1,4,5,8]
type(test_array[3])     # 인덱스[3] 값
print(test_array.dtype)  # 데이터 타입      .dtype
print(test_array.shape)  # (4,)  -> 1차원 array 에 element 4개가 있음, shape(차원구성)


# - numpy는 np.array 함수를 활용하여 배열을 생성함 -> ndarray
# - numpy는 하나의 데이터 타입만 배열에 넣을 수 있음
# - List와 가장 큰 차이점, Dynamic typing(예, [1, 2, “5”, 4.2]) not supported
# - C의 Array를 사용하여 배열을 생성함

# ## 3. Array shape 

# #### 3.1 Vector (1차원)

# In[13]:


test_array = np.array([1,4,5,8], float) 
print(test_array)
# shape은 (4, ) : 1차원에 4개의 element가 있는 벡터


# #### 3.2 Matrix (2차원)

# In[15]:


matrix = [[1,2,5,8], [2,3,4,9], [4,5,6,7]]
np.array(matrix, int).shape
# shape은 (3,4) : 행이 3개, 열이 4개인 매트릭스


# #### 3.3 Tensor (3차원)

# In[20]:


tensor = [[[1,2,5,8], [2,3,4,9], [4,5,6,7]],
          [[1,2,5,8], [2,3,4,9], [4,5,6,7]],
          [[1,2,5,8], [2,3,4,9], [4,5,6,7]],
          [[1,2,5,8], [2,3,4,9], [4,5,6,7]]]
np.array(tensor, int).shape
# shape(4,3,4) : 평면이 4개, 행이3개, 열이 4개인 텐서


# ####  3.4  ndim & size

# In[21]:


np.array(tensor, int).ndim     # 3, number of dimension
np.array(tensor, int).size     # 48


# #### 3.5 dtype

# In[22]:


np.array([[1,2,3], [4.5, "5", "6"]], dtype = np.float32)


# - Single element가 가지는 데이터 타입
# -  C의 데이터 타입과 호환
# -  nbytes – ndarray object의 메모리 크기를 바이트 단위로 반환함

# In[23]:


np.array([[1,2,3], [4.5, "5", "6"]], dtype = np.float32).nbytes    # 메모리크기 반환


# #### 3.6 reshape

# In[25]:


test_matrix = [[1,2,3,4,], [5,6,7,8]]
np.array(test_matrix).shape


# In[26]:


np.array(test_matrix).reshape(8,)    # array([1,2,3,4,5,6,7,8])


# - Array의 shape을 변경함 (element의 개수는 동일)
# - Array의 size만 같다면 다차원으로 자유로이 변형가능

# In[ ]:


np.array(test_matrix).reshape(-1, 2).shape     
# 열의 개수 2개, 행의갯수 -1 -> 열의개수를 2개를 맞추고 행의갯수는 자동으로 맞춰줘


# In[27]:


np.array(test_matrix).reshape(2, 2, 2).shape


# #### 3.7  flatten

# In[34]:


test_matrix = [[[1,2,3,4], [5,6,7,8]], [[2,3,4,5], [6,7,8,9]]]
np.array(test_matrix).shape    # (2, 2, 4)


# In[33]:


np.array(test_matrix).flatten()
np.array(test_matrix).flatten().shape # 다차원 array를 1차원 array로 변환


# - 다차원 array를 1차원 array로 변환

# ## 4. Indexing & slicing

# #### 4.1  Indexin

# In[35]:


a = np.array([[1,2,3], [4,5,6]], int)
print(a)
print(a[0,0]) # 2차원 배열 표기법 1, 기존의 슬라이싱[0],[0] - > [0,0] 가능
print(a[0][0]) # 2차원 배열 표기법 2   


# #### 4.2 Slicing

# In[43]:


a = np.array([[1,2,3,4,5], [6,7,8,9,10]], int)
print(a[:, 2:])      # , 부분 
print(a[1, 1:3])     # row 1의 1~2열
print(a[1:3])        # 1 row ~ 2 row 전체, column은 무시)
print(a[:, ::2])     # step 가능


# - List와 달리 행과 열 부분을 나눠서 slicing이 가능함
# -  Matrix의 부분 집합을 추출할 때 유용함

# ## 5.  Creation function

# #### 5.1 arange

# In[44]:


np.arange(10)     #arange - List 의 range와 같은 효과 


# In[46]:


np.arange(0,5,0.5)     # flaoting point도 표시가능


# In[48]:


np.arange(0,5,0.5).tolist()      # tolist로 list로 만들 수 있음


# In[49]:


np.arange(30).reshape(5, 6) # 0~29, 2차원 리스트 생성(행 : 6, 열 : 6)


# - List와 달리 행과 열 부분을 나눠서 slicing이 가능함
# - Matrix의 부분 집합을 추출할 때 유용함

# #### 5.2 ones, zeros and empty

# In[50]:


np.zeros(shape=(10,), dtype=np.int8)     # 원소가 10개인 벡터 생성,


# In[56]:


np.ones((2, 5))     # dtype 을 설정 안하면 1.  의 소수점으로 표시
np.ones((2, 5)).nbytes      # 실수는 double이 기본으로 설정되어서 80byte


# In[57]:


np.empty((3,5))     # # 메모리가 초기화되어 있지 않음, 공간만 비워둬라?


# - empty – shape만 주어지고 비어있는 ndarray 생성

# #### 5.3 Something like

# In[61]:


test_matrix = np.arange(30).reshape(5,6)
print(test_matrix)
np.ones_like(test_matrix)     # 기존의 0~29까지 value를 1로 채움
np.zeros_like(test_matrix)    # 기존의 0~29까지의 vlaue를 0으로 채움


# - 기존 ndarray의 shape 크기 만큼 1, 0 또는 empty array를 반환

# ####  5.4 identity (단위 행렬 생성)

# In[62]:


np.identity(n=3, dtype = np.int8)


# In[65]:


np.identity(5)      # 기본이 double이라 실수로 표현됨 


# #### 5.5 eye(대각선이 1인 행렬)

# In[66]:


np.eye(N=3, M=5, dtype=np.int8)     # M 은 열의 개수


# In[67]:


np.eye(5)


# In[68]:


np.eye(3, 5, k=2)      # k는 시작 인덱스


# - N은 행렬의 개수, M은 열의 개수, K는 시작인덱스의 위치 

# #### 5.6 diag (대각 행렬의 값을 추출)

# In[74]:


matrix = np.arange(9).reshape(3,3)
print(matrix)
np.diag(matrix)
np.diag(matrix, k=1)       # k -> start 인덱스 


# #### 5.7 Random sampling

# In[76]:


np.random.seed(seed = 1000)      # .seed로 난수 생성 초기값 지정


# In[77]:


np.random.uniform(0,1,10).reshape(2,5)     # uniform 균등분포


# In[78]:


np.random.normal(0, 1, 10).reshape(2,5)      # normal 정규분포, 평균이 0, 표준편차1인 정규분포


# In[80]:


# np.random.binomial(n, p, size) # 이항 분포
# np.random.poisson(lam. size) # 포아송 분포
# np.random.standard_t(df, size) # t-분포
# np.random.f(dfnum, dfden, size) # F-분포


# ## 6. Operation Function

# #### 6.1 Sum

# In[84]:


test_array = np.arange(1,11)
test_array.sum(dtype = np.float)     # 


# #### 6.2 Axis

# - 모든 operation function을 실행할 때, 기준이 되는 dimension 축

# In[85]:


test_array = np.arange(1,13).reshape(3,4)
print(test_array)


# In[87]:


test_array.sum(axis=1)     # axis로 방향설정 가능 axis = 1 -> 행 단위 sum 실행


# In[88]:


test_array.sum(axis=0)     # axis로 방향설정 가능 axis = 0 -> 열 단위 sum 실행


# #### 6.3 mean & std

# In[94]:


test_array = np.arange(1,13).reshape(3,4)
test_array.mean()
test_array.mean(axis = 1)    # 행 기준 mean 실행
test_array.mean(axis = 0)


# #### 6.4 Mathematical functions(다양한 수학함수 사용)

# - 지수 함수: exp, expml, exp2, log, log10, loglp, log2, power, sqrt
# - 삼각 함수: sin, cos, tan, arcsin, arccos, arctan
# - Hyperbolic: sinh, cosh, tanh, arcsinh, arccosh, arctanh

# In[97]:


print(np.exp(test_array))
print(np.sqrt(test_array))


# #### 6.5 Concatenate (Numpy array를 합치는 함수)

# In[117]:


a = np.array([[1,2], [3,4]])
print(a)
b = np.array([5,6])
print(b)
c = np.vstack((a,b))      # vstack  a에 b를 쌓는 개념
print(c)


# In[118]:


a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a,b), axis=0) 


# In[110]:


a = np.array([[1], [2], [3]])
b = np.array([[2], [3], [4]])
np.hstack((a,b))      # 세로로 쌓는다?


# In[112]:


a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a,b.T), axis=1)      


# ## 7. Array operation

# #### 7.1 Operations btw arrays (기본적인 사칙 연산 지원)

# In[121]:


test_a = np.array([[1,2,3], [4,5,6]], float)
print(test_a + test_a)      # 덧셈 연산
print(test_a * test_a)      # 곱셈 연산


# #### 7.2 Dot product
# - 매트릭스 곱셈     # (l,m) x (m,n)  (l,n)

# In[127]:


test_a = np.arange(1,7).reshape(2,3)
test_b = np.arange(7,13).reshape(3,2)
print(test_a)
print(test_b)
test_a.dot(test_b)


# #### 7.3 Transpose

# In[131]:


test_a = np.arange(1,7).reshape(2,3)
test_a.transpose()      # 행 :  2 , 열 : 3인 매트릭스를 행 3, 열 2로 뒤집어 바꿈
test_a.T


# #### 7.4 Broadcasting (Shape이 다른 배열간 연산 지원)

# In[133]:


test_matrix = np.array([[1,2,3], [4,5,6]], float)
scalar = 3
test_matrix + scalar       # 각 element에 scalar 값을 더함


# In[135]:


test_matrix - scalar   # 덧셈
test_matrix * scalar   # 뺄셈
test_matrix / scalar   # 나누기
test_matrix // scalar  # 몫
test_matrix ** 2       # 제곱 


# In[138]:


# Matrix와 Venctor 간의 연산도 가능하다
test_matrix = np.arange(1,13).reshape(4,3)
test_vector = np.arange(10,40,10)
print(test_matrix)
print(test_vector)

test_matrix + test_vector 


# ## 8. Comparison

# #### 8.1 All & Any

# In[145]:


a = np.arange(10)
print(a)
print(np.any(a>5))     # any는 하나라도 true가 있을 시 True로 표시
print(np.any(a<0))     # any는 하나라도 False가 있을 시 False로 표시


# In[146]:


print(np.all(a>5))  # all – 모두가 조건을 만족해야 True
print(np.all(a<10)) # 


# In[147]:


# 배열의 크기가 동일할때 원소간 비교 가능
test_a = np.array([1, 3, 0], float)
test_b = np.array([5, 2, 1], float)
test_a > test_b 


# In[148]:


test_a == test_b


# In[150]:


(test_a > test_b).any()


# #### 8.2 Logical operation

# In[151]:


a = np.array([1, 3, 0], float)
b = np.logical_and(a>0, a<3)      # and는 조건
print(b)


# In[154]:


c = np.logical_not(b)
print(c)


# In[156]:


np.logical_or(b,c)


# In[158]:


np.where(a>0, 3, 2)      # where(condition, True, False)


# In[160]:


a = np.arange(10, 20)
np.where(a>15)


# In[163]:


a = np.array([1, np.NaN, np.Inf], float)     # np.NAN -> not a Number
print(a)


# In[164]:


np.isnan(a)     # is Not a Number?


# In[165]:


np.isfinite(a)      # is finite number?


# #### 8. argmax & argmin (array내 최대값 또는 최소값의 index를 리턴)

# In[166]:


a = np.array([1,2,4,5,8,78,23,3])
np.argmax(a), np.argmin(a)      # argmax - 최댓값, argmin - 최솟값


# In[169]:


a = np.array([[1,2,4,7],[9,88,6,45],[8,78,23,3]])
print(a)


# In[172]:


np.argmax(a, axis=1)      # 출력은 value값이 아닌 index로 표시


# In[174]:


np.argmax(a, axis=0)


# ## 9. Boolean & fancy index

# #### 9.1 Boolean index

# In[178]:


test_array = np.array([1,4,0,2,3,8,9,7], float)
print(test_array)
test_array > 3


# In[177]:


test_array[test_array > 3]     # [] 안에 조건 설정 후 출력, 해당되는 것만 출력됨


# In[179]:


condition = test_array < 3     # 조건 자체를 지정 후 출력 [condition]
test_array[condition]


# #### 9.2 Fancy index

# In[180]:


a = np.array([2, 4, 6, 8], float)
b = np.array([0, 0, 1, 3, 2, 1], int)      # 반드시 integer로 선언
a[b]       # b 배열의 값을 인덱스로 하여 a의 값들을 출력


# In[182]:


a = np.array([[1,4], [9,16]], float)
b = np.array([0,0,1,1,1], int)
c = np.array([0,1,1,1,0], int)
a[b,c]       # b를 row index, c를 column index로 변환하여 표시


# ## 10. Numpy data I/O

# #### 10.1 loadtxt & savetxt (Text type의 데이터를 읽고 저장하는 기능)

# In[187]:


a = np.loadtxt('hi.txt') 
a[:10]


# In[204]:


a_int = a.astype(int)     # a.astype(int) a를 int로 변환
a_int[:3]     


# In[206]:


# np.savetxt('number.csv', a_int,delimiter=‘,’)   -> csv로 저장


# #### 10.2 numpy object – npy
# - Numpy object(pickle) 형태로 데이터를 저장하고 불러옴
# - Binary 파일 형태
# ##### np.save(‘npy_test’, arr=a_int)
# ##### npy_array = np.load(file=‘npy_test.npy’)

# #### 연습문제 

# In[251]:


# 1. 길이가 10인 0-벡터를 만드세요 
np.zeros(10)


# In[252]:


# 2. 길이가 10이며 다섯번째 원소만 1이고 나머지 원소는 모두 0인 벡터를 만드세요.
a = np.where(np.arange(10)==4, 1, 0)
print(a)
b = np.eye(1, 10, k=4)
print(a)


# In[256]:


# 3. 10 부터 49까지의 값을 가지는 벡터를 만드세요.
a = np.arange(10,50)
print(a)


# In[257]:


# 4. 위(3번) 벡터의 순서를 바꾸세요.
c = np.arange(49, 9, -1)
print(c)


# In[258]:


# 5. 0부터 8까지의 값을 가지는 3x3 행렬을 만드세요.
np.arange(0,9).reshape(3,3)


# In[259]:


# 6. 벡터 [1,2,0,0,4,0] 에서 원소의 값이 0이 아닌 원소만 선택한 벡터를 만드세요.
a = np.array([1,2,0,0,4,0], int)
a[a != 0]


# In[261]:


# 7. 3x3 단위 행렬(identity matrix)을 만드세요
b = np.identity(n=3, dtype = np.int8)
print(b)


# In[273]:


# 8. 난수 원소를 가지는 3x3 행렬을 만드세요
d = np.random.random(9).reshape(3,3)
print(d)


# In[275]:


# 9. 위(8번)에서 만든 난수 행렬에서 최대값/최소값 원소를 찾으세요
d[np.argmax(d) // 3, np.argmax(d) % 3]


# In[276]:


#9 최솟값
d[np.argmin(d) // 3, np.argmin(d) % 3]


# In[277]:


# 10. 위(8번)에서 만든 난수 행렬에서 행 평균, 열 평균을 계산하세요
np.mean(d, axis = 1)


# In[278]:


# # 10. 위(8번)에서 만든 난수 행렬에서 행 평균, 열 평균을 계산하세요
np.mean(d, axis = 0)

