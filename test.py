import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

url = "https://vnexpress.net/cong-bo-duong-day-nong-ho-tro-160-thuyen-vien-viet-nam-o-trung-dong-5046145.html"

dict = {"a": "b"}

url_link = next(iter(dict.keys()))

response  = requests.get(url_link, headers = headers)