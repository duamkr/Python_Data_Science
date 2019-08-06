# 연습문제 1
# key1의 값을 기준으로 data1의 값을 분류하여 합계를 구한 결과를 시리즈가 아닌 데이터프레임으로 구한다.

df2 = pd.DataFrame({
    'key1': ['A', 'A', 'B', 'B', 'A'],
    'key2': ['one', 'two', 'one', 'two', 'one'],
    'data1': [1, 2, 3, 4, 5],
    'data2': [10, 20, 30, 40, 50]
})
df2

a =pd.DataFrame(df2.data1.groupby(df2.key1).sum())
print(a)

# 연습 문제 2
# 붓꽃(iris) 데이터에서 붓꽃 종(species)별로 꽃잎길이(sepal_length), 꽃잎폭(sepal_width) 등의 평균을 구하라. 만약 붓꽃 종(species)이 표시되지 않았을 때 이 수치들을 이용하여 붓꽃 종을 찾아낼 수 있을지 생각하라.

import seaborn as sns
iris = sns.load_dataset("iris")
iris

def flower(a) :
    return a.mean()

iris.groupby(iris.species).apply(flower)

# 연습 문제 3
# 1. 팁의 비율이 요일과 점심/저녁 여부, 인원수에 어떤 영향을 받는지 살펴본다.
# 2. 어떤 요인이 가장 크게 작용하는지 판단할 수 있는 방법이 있는가?

import seaborn as sns
tips = sns.load_dataset("tips")
tips.tail()

tips["tip_pct"] = tips['tip'] / tips['total_bill']
tips.head()
tips.columns
# 1.
tips.pivot_table('tip_pct', index = ['day','time','size'])

# 2.
tips.columns
tips.pivot_table('tip_pct', index = ['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size'])
a.sort_values(by =['tip_pct'])
a.head()