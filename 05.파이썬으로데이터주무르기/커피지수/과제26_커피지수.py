from bs4 import BeautifulSoup as bs
from pprint import pprint
import requests
import pandas as pd


html = requests.get('https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=1&sido=&gugun=&store=')
soup = bs(html.text,'html.parser')
html.close()

# result = soup.select('td a')
result = soup.find_all('td', 'noline center_t')

a = []
b = []
for r in result:
    a = r.text
    b.append([a])
    print(b)

a = []
b = []
for i in range(1,57) :
    html = 'https://www.hollys.co.kr/store/korea/korStore2.do?pageNo='
    html2 = '&sido=&gugun=&store='
    html3 = requests.get(html + str(i) + html2)
    soup = bs(html3.text, 'html.parser')
    result = soup.find_all('td', 'noline center_t')

    for r in result:
        a = r.text
        b.append([a])

hollys_coffee = pd.DataFrame({'address': b})
hollys_coffee.head()

hollys_coffee.to_excel("test.xlsx")



ediya_html = requests.get('https://ediya.com/contents/find_store.html#c')
ediya_soup = bs(ediya_html.text,'html.parser')
ediya_result = ediya_soup.find_all('div','st_info_con')
d =ediya_soup.select('st_info_con p')