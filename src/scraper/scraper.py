import requests
from bs4 import BeautifulSoup
import time
import json
import os

#chuyên mục

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

base_url = "https://vnexpress.net"

def get_articles_links(category, base_url=base_url, max_pages=1):
    articles_links = {}
    if category==None:
        category='tin-tuc-24h'
    
    for page in range(max_pages):
        
        url = f"{base_url}/{category}" if page==0 else f"{base_url}/{category}-p{page+1}"
        response  = requests.get(url, headers = headers)
        soup = BeautifulSoup(response.content, "html.parser")
        news_title = soup.find_all(["h3", "h4"], class_="title-news")

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

    title_tag = soup.find('h1', class_='title-detail')
    title = title_tag.get_text(strip=True) if title_tag else ""
    description_tag = soup.find('p', class_="description")
    description = description_tag.get_text(strip=True) if description_tag else ""
    contents_raw = soup.find_all('p', class_="Normal") 
    for con in contents_raw[:-1]:
        contents.append(con.text.strip())
    contents_string = "\n\n".join(p.strip() for p in contents)
    time_tag = soup.find("span", class_="date")
    time = time_tag.get_text(strip=True) if time_tag else ""
    author = contents_raw[-1].text.strip() if contents_raw else ""

    if contents_raw != "":
        return {
            "url": url_link,
            "title": title,
            "description": description,
            "date": time,
            "content": contents_string,
            "author": author
        }
    else:
        pass

def save_articles_json(category='tin-tuc-24h'):
    category_path = category.replace("-","_")
    output_path=f"data/articles_{category_path}.json"

    if os.path.exists(output_path):
        os.remove(output_path)

    os.makedirs("data", exist_ok=True)

    all_articles=[]

    articles_links = get_articles_links(category, max_pages=1)
    for article in articles_links.values():
        article_json = crawl_article(article)

        if article_json:
            article_json["category"] = category
            if article_json["content"] != "":
                all_articles.append(article_json)
        time.sleep(1)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    return all_articles, output_path



if __name__ == "__main__":
    save_articles_json()
            

