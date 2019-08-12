
from bs4 import BeautifulSoup as bs
from pprint import pprint
import requests
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'Malgun Gothic'
plt.rcParams["font.size"] = 10

import os
currentpath = os.getcwd()
os.chdir('E:/workspace/Python_Data_Science/05.파이썬으로데이터주무르기/커피지수')

print("hello world")
total_coffee = pd.read_excel('total_coffee.xlsx', header = 0)
total_coffee.head()
total_coffee["coffee_index"] = total_coffee['bback'] + total_coffee['starbucks'] + total_coffee['bean'] / total_coffee['Ediya']

total_coffee['시도'].unique()

si_name  = [None] * len(total_coffee)

tmp_gu_dict = {'수원':['장안구', '권선구', '팔달구', '영통구'],
               '성남':['수정구', '중원구', '분당구'],
               '안양':['만안구', '동안구'],
               '안산':['상록구', '단원구'],
               '고양':['덕양구', '일산동구', '일산서구'],
               '용인':['처인구', '기흥구', '수지구'],
               '청주':['상당구', '서원구', '흥덕구', '청원구'],
               '천안':['동남구', '서북구'],
               '전주':['완산구', '덕진구'],
               '포항':['남구', '북구'],
               '창원':['의창구', '성산구', '진해구', '마산합포구', '마산회원구'],
               '부천':['오정구', '원미구', '소사구']}

for n in total_coffee.index:
    if total_coffee['광역시도'][n][-3:] not in ['광역시', '특별시', '자치시']:
        if total_coffee['시도'][n][:-1] == '고성' and total_coffee['광역시도'][n] == '강원도':
            si_name[n] = '고성(강원)'
        elif total_coffee['시도'][n][:-1] == '고성' and total_coffee['광역시도'][n] == '경상남도':
            si_name[n] = '고성(경남)'
        else:
            si_name[n] = total_coffee['시도'][n][:-1]

        for keys, values in tmp_gu_dict.items():
            if total_coffee['시도'][n] in values:
                if len(total_coffee['시도'][n]) == 2:
                    si_name[n] = keys + ' ' + total_coffee['시도'][n]
                elif total_coffee['시도'][n] in ['마산합포구', '마산회원구']:
                    si_name[n] = keys + ' ' + total_coffee['시도'][n][2:-1]
                else:
                    si_name[n] = keys + ' ' + total_coffee['시도'][n][:-1]

    elif total_coffee['광역시도'][n] == '세종특별자치시':
        si_name[n] = '세종'

    else:
        if len(total_coffee['시도'][n]) == 2:
            si_name[n] = total_coffee['광역시도'][n][:2] + ' ' + total_coffee['시도'][n]
        else:
            si_name[n] = total_coffee['광역시도'][n][:2] + ' ' + total_coffee['시도'][n][:-1]

si_name
total_coffee["ID"] = si_name

total_coffee.head()

# Cartogram으로 우리나라 지도 만들기

draw_korea_raw = pd.read_excel('05. draw_korea_raw.xlsx')
draw_korea_raw_stacked = pd.DataFrame(draw_korea_raw.stack())
draw_korea_raw_stacked.reset_index(inplace=True)
draw_korea_raw_stacked.rename(columns={'level_0':'y', 'level_1':'x', 0:'ID'},
                              inplace=True)

draw_korea_raw_stacked
draw_korea = draw_korea_raw_stacked

# 변수 이름 변경

BORDER_LINES = [
    [(5, 1), (5,2), (7,2), (7,3), (11,3), (11,0)], # 인천
    [(5,4), (5,5), (2,5), (2,7), (4,7), (4,9), (7,9),
     (7,7), (9,7), (9,5), (10,5), (10,4), (5,4)], # 서울
    [(1,7), (1,8), (3,8), (3,10), (10,10), (10,7),
     (12,7), (12,6), (11,6), (11,5), (12, 5), (12,4),
     (11,4), (11,3)], # 경기도
    [(8,10), (8,11), (6,11), (6,12)], # 강원도
    [(12,5), (13,5), (13,4), (14,4), (14,5), (15,5),
     (15,4), (16,4), (16,2)], # 충청북도
    [(16,4), (17,4), (17,5), (16,5), (16,6), (19,6),
     (19,5), (20,5), (20,4), (21,4), (21,3), (19,3), (19,1)], # 전라북도
    [(13,5), (13,6), (16,6)], # 대전시
    [(13,5), (14,5)], #세종시
    [(21,2), (21,3), (22,3), (22,4), (24,4), (24,2), (21,2)], #광주
    [(20,5), (21,5), (21,6), (23,6)], #전라남도
    [(10,8), (12,8), (12,9), (14,9), (14,8), (16,8), (16,6)], #충청북도
    [(14,9), (14,11), (14,12), (13,12), (13,13)], #경상북도
    [(15,8), (17,8), (17,10), (16,10), (16,11), (14,11)], #대구
    [(17,9), (18,9), (18,8), (19,8), (19,9), (20,9), (20,10), (21,10)], #부산
    [(16,11), (16,13)], #울산
#     [(9,14), (9,15)],
    [(27,5), (27,6), (25,6)],
]

# 혜식님의 코드
plt.figure(figsize=(8, 11))

# 지역 이름 표시
for idx, row in draw_korea.iterrows():

    # 광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시한다.
    # (중구, 서구)
    if len(row['ID'].split()) == 2:
        dispname = '{}\n{}'.format(row['ID'].split()[0], row['ID'].split()[1])
    elif row['ID'][:2] == '고성':
        dispname = '고성'
    else:
        dispname = row['ID']

    # 서대문구, 서귀포시 같이 이름이 3자 이상인 경우에 작은 글자로 표시한다.
    if len(dispname.splitlines()[-1]) >= 3:
        fontsize, linespacing = 9.5, 1.5
    else:
        fontsize, linespacing = 11, 1.2

    plt.annotate(dispname, (row['x'] + 0.5, row['y'] + 0.5), weight='bold',
                 fontsize=fontsize, ha='center', va='center',
                 linespacing=linespacing)

# 시도 경계 그린다.
for path in BORDER_LINES:
    ys, xs = zip(*path)
    plt.plot(xs, ys, c='black', lw=1.5)

plt.gca().invert_yaxis()
# plt.gca().set_aspect(1)

plt.axis('off')

plt.tight_layout()
plt.show()


# total_coffee 와 draw_coffee의 확인 비교 (명칭)
set(draw_korea['ID'].unique()) - set(total_coffee['ID'].unique())
set(total_coffee['ID'].unique()) - set(draw_korea['ID'].unique())

tmp_list = list(set(total_coffee['ID'].unique()) - set(draw_korea['ID'].unique()))

for tmp in tmp_list:
    total_coffee = total_coffee.drop(total_coffee[total_coffee['ID'] == tmp].index)

print(set(total_coffee['ID'].unique()) - set(draw_korea['ID'].unique()))
total_coffee.head()

# key값으로 total_coffee와 draw_korea merge
total_coffee = pd.merge(total_coffee, draw_korea, how='left', on=['ID'])

total_coffee.head()

mapdata = total_coffee.pivot_table(index='y', columns='x', values='coffee_index')
masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)

# 함수 지정

def drawKorea(targetData, blockedMap, cmapname):
    gamma = 0.75

    whitelabelmin = (max(blockedMap[targetData]) -
                     min(blockedMap[targetData])) * 0.25 + \
                    min(blockedMap[targetData])

    datalabel = targetData

    vmin = min(blockedMap[targetData])
    vmax = max(blockedMap[targetData])

    mapdata = blockedMap.pivot_table(index='y', columns='x', values=targetData)
    masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)

    plt.figure(figsize=(9, 11))
    plt.pcolor(masked_mapdata, vmin=vmin, vmax=vmax, cmap=cmapname,
               edgecolor='#aaaaaa', linewidth=0.5)

    # 지역 이름 표시
    for idx, row in blockedMap.iterrows():
        # 광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시한다.
        # (중구, 서구)
        if len(row['ID'].split()) == 2:
            dispname = '{}\n{}'.format(row['ID'].split()[0], row['ID'].split()[1])
        elif row['ID'][:2] == '고성':
            dispname = '고성'
        else:
            dispname = row['ID']

        # 서대문구, 서귀포시 같이 이름이 3자 이상인 경우에 작은 글자로 표시한다.
        if len(dispname.splitlines()[-1]) >= 3:
            fontsize, linespacing = 10.0, 1.1
        else:
            fontsize, linespacing = 11, 1.

        annocolor = 'white' if row[targetData] > whitelabelmin else 'black'
        plt.annotate(dispname, (row['x'] + 0.5, row['y'] + 0.5), weight='bold',
                     fontsize=fontsize, ha='center', va='center', color=annocolor,
                     linespacing=linespacing)

    # 시도 경계 그린다.
    for path in BORDER_LINES:
        ys, xs = zip(*path)
        plt.plot(xs, ys, c='black', lw=2)

    plt.gca().invert_yaxis()

    plt.axis('off')

    cb = plt.colorbar(shrink=.1, aspect=10)
    cb.set_label(datalabel)

    plt.tight_layout()
    plt.show()

# 커피지수 확인하기
drawKorea('coffee_index', total_coffee, 'Blues')