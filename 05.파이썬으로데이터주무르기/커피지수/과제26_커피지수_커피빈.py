from bs4 import BeautifulSoup as bs
from pprint import pprint
import requests
import pandas as pd



html = requests.get("STORE _ COFFEE BEAN KOREA.html",'r').read()
soup = bs(html.text,'html.parser')
result = soup.find('p','address')
abcde = soup.find_all('p','address')


for ul in soup.find_all('p','address') :
    a = ul.text
    print(a)



a = []
b = []
for r in result2:
    a = r.text
    print(a)