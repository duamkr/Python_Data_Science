import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.size"] = 10

import os
currentpath = os.getcwd()
os.chdir('D:/workspace/Python_Data_Science/08.시험_190814')


# 1. 다음과 같은 어레이를 Numpy를 이용하여 만드시오
# 1)
a = np.arange(0,5,0.5)
# 2)
b = np.arange(1,11).reshape((2,5))
# 3)
c = np.identity(n=3, dtype=np.int8)
# 4)
d = np.ones((4,4),dtype=np.int32)

# 5)
e = np.arange(1,25,2).reshape(3,4)



# 2. 다음과 같은 데이터 프레임을 만드시오

data = {
    "2015": [9904312, 3448737, 2890451, 2466052],
    "2010": [9631482, 3393191, 2632035, 2431774],
    "2005": [9762546, 3512547, 2517680, 2456016],
    "2000": [9853972, 3655437, 2466338, 2473990],
    "지역": ["수도권", "경상권", "수도권", "경상권"]
}
columns = ["지역", "2015", "2010", "2005", "2000"]
index = ["서울", "부산", "인천", "대구"]
df = pd.DataFrame(data, index=index, columns=columns)
df["2010-2015 증가율"] = ((df["2015"] - df["2010"]) / df["2010"]).round(4)

# 3. 타이타닉호 슥액에 대해서 다음을 구하시오
import seaborn as sns
titanic = sns.load_dataset("titanic")

# 1) 성별(sex) 인원수, 선실수(class) 인원수, 사망/생존(alive) 인원수를 구하시오
titanic['sex'].value_counts()
titanic['class'].value_counts()
titanic['alive'].value_counts()

# 2) 미성년자, 청년, 장년, 노년 승객의 비율을 구하시오(단, 나이의 기준은 1,15,30,45,60,99)
alive_titanic = titanic[titanic["alive"] == "yes"]
dead_titanic = titanic[titanic["alive"] == "no"]

bins = [1, 15, 25, 35, 60, 99]
labels = ["미성년자", "청년", "중년", "장년", "노년"]

alive_titanic_cat = pd.cut(alive_titanic["age"],bins, labels = labels)
dead_titanic_cat = pd.cut(dead_titanic["age"],bins, labels = labels)

alive_titanic_cat.value_counts()
live1 = ((alive_titanic_cat.value_counts() / 278 ) * 100).round(1)
print(live1)

dead_titanic_cat.value_counts()
dead1 = ((dead_titanic_cat.value_counts() / 422 ) * 100).round(1)
print(dead1)

# 4. 팁 데이터에 대해서 다음을 구하시오.
tips = sns.load_dataset("tips")
tips.head()
# 1) 팁의 비율(단위 %)을 소숫점 2째짜리까지 구하시오.
tips["tip_pct"] = ((tips['tip'] / tips['total_bill']) * 100).round(2)

# 2) 팁의 비율이 가장 높은 날은 목, 금, 토, 일요일 중 어떤 날인지 피봇테이블을 이용하여 구하시오
tips.pivot_table('tip_pct', index = ['day']).round(2)
# 금요일이 16.99% 로 가장 높음

# 6. 시그모이드 함수 (sgmoid(x) = 1 / (1 + np.exp(-x)))와 시그모이드 함수를 미분한 함수
# 의 그래프를 그리시오 (단, x의 범위는 -3에서 +3까지)
def sigmoid(x) :
    return  1 / (1 + np.exp(-x))

x = np.arange(-3,4,1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()


# 7. 다음의 빈칸에 들어갈 말은?
# 1). 경사하강법
# 2). 기울기
# 3). 퍼셉트론
# 4). relu
# 5). 은닉층

# 8. 아이리스 데이터 셋을 이용하여 다음을 구하는 프로그램을 작성하시오
# - 아이리스 데이터의 4가지 속성(꽃받침 길이/ 폭, 꽃잎 길이/폭)을 이용하여 품종을 예측
# - 단 정확도는 98% 이상일 것

iris = sns.load_dataset("iris")
iris.head()

from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = iris
sns.pairplot(df, hue='species');
plt.show()

# 데이터 분류
dataset = df.values
X = dataset[:,0:4].astype(float)
Y_1 = dataset[:,4]

# Y_1의 문자열을 숫자로 변경
e = LabelEncoder()
e.fit(Y_1)
Y = e.transform(Y_1)
Y = np_utils.to_categorical(Y)

# 모델 설정
model = Sequential()
model.add(Dense(16, input_dim = 4, activation= 'relu'))
model.add(Dense(3, activation= 'softmax'))

# 모델 컴파일

model.compile(loss = 'categorical_crossentropy',
              optimizer= 'adam',
              metrics= ['accuracy'])

# 모델 실행
model.fit(X, Y, epochs= 100, batch_size= 2)

# 출력
print("\n Accuracy : %.4f" % (model.evaluate(X, Y)[1]))

# 98.67 %

# 9. 피마 인디언 데이터 셋을 이용하여 다음을 구하는 프로그램을 작성하시오
# - 피마인디언 데이터의 8가지 속성을 이용하여 당뇨병 여부를 판단할것
# - 단, 데이터의 25%는 테스트 데이터로 사용하여 정확도를 구할 것

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 불러오기
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter = ",")
X1 = dataset[:,0:8]
Y1 = dataset[:,8]

# 학습셋과 테스트셋 구분
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size = 0.25, random_state=seed)

# 모델 설정
model = Sequential()
model.add(Dense(40, input_dim = 8, activation  = 'relu'))
model.add(Dense(24, activation= 'relu'))
model.add(Dense(6, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

# 모델 컴파일

model.compile(loss = 'binary_crossentropy',
              optimizer= 'adam',
              metrics = ['accuracy'])

# 모델 실행
model.fit(X_train,Y_train, epochs= 1000, batch_size= 50)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X_test,Y_test)[1]))