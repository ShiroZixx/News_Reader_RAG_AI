import requests
from bs4 import BeautifulSoup
import time
import json
import os

os.makedirs("data/raw", exist_ok=True)

#chuyên mục
categories = ["tin-tuc-24h", "thoi-su", "the-gioi", "kinh-doanh", "khoa-hoc-cong-nghe", 
              "goc-nhin", "spotlight", "bat-dong-san", "suc-khoe", "giai-tri", 
              "the-thao", "phap-luat", "giao-duc", "doi-song", "xe", "du-lich",
              "anh", "infographic", "y-kien", "tam-su", "cuoi"]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

base_url = "https://vnexpress.net"

def get_article_links(category, max_pages=1):
    articles_links = {}
    
    for page in range(max_pages):
        
        url = f"{base_url}/{category}" if page==0 else f"{base_url}/{category}-p{page+1}"
        response  = requests.get(url, headers = headers)
        soup = BeautifulSoup(response.content, "html.parser")
        news_title = soup.find_all("h3", class_="title-news")

        for new in news_title:
            titles = new.find("a")

            if titles:
                title_text = titles.get_text(strip=True)
                title_url = titles["href"]

                articles_links[title_text] = title_url

    return articles_links


def crawl_article(articles_link):
    contents = []

    url_link = articles_link

    response  = requests.get(url_link, headers = headers)
    soup = BeautifulSoup(response.content, "html.parser")

    title = soup.find('h1', class_='title-detail').text.strip()
    description = soup.find('p', class_="description").text.strip()
    contents_raw = soup.find_all('p', class_="Normal")
    for con in contents_raw[:-1]:
        contents.append(con.text.strip())
    contents_string = "\n".join(p.strip() for p in contents)
    time = soup.find("span", class_="date").text.strip()
    author = contents_raw[-1].text.strip()

    return {
        "url": url_link,
        "title": title,
        "description": description,
        "date": time,
        "content": contents_string,
        "author": author
    }

def crawl_one_category(output_path="data/articles.json", category='tin-tuc-24h'):
    all_articles=[]

    article_links = get_article_links(category, max_pages=1)
    for article in article_links.values():
        article_json = crawl_article(article)

        if article_json:
            article_json["category"] = category
            all_articles.append(article_json)
        time.sleep(1)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    return all_articles

#In future
def crawl_all_categories():
    pass


if __name__ == "__main__":
    crawl_one_category()
            



            





    








    #print(response.content)
