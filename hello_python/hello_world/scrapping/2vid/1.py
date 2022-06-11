import requests 
import fake_useragent
from bs4 import BeautifulSoup

user = fake_useragent.UserAgent().random
header = {'user-agent': user}

link = "https://alar-q.github.io/Amazing_Front/"

responce = requests.get(link, headers=header).text

soup = BeautifulSoup(responce, "lxml")

block = soup.find("div", id="karlsson2").text # also we have find_all

print(block)
