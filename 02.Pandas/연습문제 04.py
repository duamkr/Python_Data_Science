# 연습 문제 4.2.5

# 다음 명령으로 타이타닉호 승객 데이터를 데이터프레임으로 읽어온다. 이 명령을 실행하려면 seaborn 패키지가 설치되어 있어야 한다.
import numpy as np
import pandas as pd
import seaborn as sns
titanic = sns.load_dataset("titanic")

# 타이타닉호 승객 데이터의 데이터 값을 각 열마다 구해본다.
print('hello')

titanic.count()




# 연습 문제 4.2.6
# 타이타닉호 승객중 성별(sex) 인원수, 나이별(age) 인원수, 선실별(class) 인원수, 사망/생존(alive) 인원수를 구하라.

titanic['sex'].value_counts()
titanic['age'].value_counts()
titanic['class'].value_counts()
titanic['alive'].value_counts()

# 연습 문제 4.2.7
bins = [1, 15, 25, 35, 60, 99]
labels = ["미성년자", "청년", "중년", "장년", "노년"]

# 타이타닉호 승객을 사망자와 생존자 그룹으로 나누고 각 그룹에 대해 '미성년자', '청년', '중년', '장년', '노년' 승객의 비율을 구한다.
# 각 그룹 별로 비율의 전체 합은 1이 되어야 한다.
titanic.head(5)

alive_titanic = titanic[titanic["alive"] == "yes"]
dead_titanic = titanic[titanic["alive"] == "no"]

bins = [1, 15, 25, 35, 60, 99]
labels = ["미성년자", "청년", "중년", "장년", "노년"]

alive_titanic_cat = pd.cut(alive_titanic["age"],bins, labels = labels)
dead_titanic_cat = pd.cut(dead_titanic["age"],bins, labels = labels)

a = ((alive_titanic_cat.value_counts() / 278 ) * 100).round(1)
alive_titanic_cat.value_counts()


dead_titanic_cat.value_counts()
d = ((dead_titanic_cat.value_counts() / 422 ) * 100).round(1)


# 살아있는 승객의 비율
print(a)

#  죽은 승객의 비율
print(d)

