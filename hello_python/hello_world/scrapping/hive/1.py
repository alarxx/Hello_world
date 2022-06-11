import requests 
import fake_useragent
from bs4 import BeautifulSoup

session = requests.Session()

user = fake_useragent.UserAgent().random

headers = {
	'user-agent': user
}

data = {
	"login": "dastanpisos@gmail.com",
	"password": "^5D=f$GiqW)?PG,",
	"remember": true,
	"twofa_code": ""
}

link = "https://the.hiveos.farm/login"

responce = requests.post(link, data=data, headers=headers).text

