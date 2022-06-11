import requests 
import fake_useragent
from bs4 import BeautifulSoup

session = requests.Session()

user = fake_useragent.UserAgent().random

link = "https://the.hiveos.farm/login"

headers = {
	'user-agent': user
}

data = {
	"login": "dastanpisos@gmail.com",
	"password": "^5D=f$GiqW)?PG,",
	"remember": true,
	"twofa_code": ""
}

responce = session.post(link, data=data, headers=headers).text

# scrap.krabov@mail.ru
# gAmD5ATjmJdvL3n

"""
{
	"username": "asdfasdf@mail.ru",
	"Login": "asdfasdf@mail.ru",
	"password": "asdfasdf",
	"Password": "asdfasdf",
	"saveauth": "1",
	"new_auth_form": "1",
	"FromAccount": "opener=account&fb=1&vk=1&ok=1&twoSteps=1",
	"act_token": "3a1027f4ceea4d6b83a66edc663272ea",
	"page": "https://my.mail.ru/?app_id_mytracker=undefined&authid=l0ve75ji.q6&dwhsplit=s10273.b1ss12743s&from=login&mt_click_id=mt-psc3a5-1647545743-2222798509&mt_sub1=other&utm_campaign=my.mail.ru&utm_medium=portal_navigation_headline&utm_source=portal",
	"lang": "ru_RU"
}
"""