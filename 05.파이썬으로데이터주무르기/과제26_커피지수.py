from bs4 import BeautifulSoup as bs
from pprint import pprint
import requests

html = requests.get('https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=1&sido=&gugun=&store=')
soup = bs(html.text,'html.parser')
html.close()

data1 = soup.find('table',{'class':'tb_store'})
data4 = soup.find_all('td',{'class':'noline center_t'}).string
data5 = soup.find('td',{'class':'noline center_t'}).string
data5 = data4.split()
data5 = data4.find.find('td', 'noline center_t').get_text()

for td in data1 :
    if td.find('td',{'class':'noline center_t'}) :
        jijum = td.find('td',{'class':'noline center_t'}).text
        data.append(jijum)
data

pprint(data1)
data2 = data1.find_all('td',{'class':'noline center_t'}).text
data3 = data1.find('td',{'class':'noline center_t'}).text
pprint(data3)



for df in soup :
    if df.find('div',{'class':'tableType01'}) :
        hollys = df.find('td',{'class':'noline center_t'}).text
        data.append([hollys])
data

