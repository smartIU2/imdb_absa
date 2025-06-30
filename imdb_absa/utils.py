from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

def get_reviews(site, title_id):

    #TODO: support different review sites

    req = Request(f"https://www.{site}.com/title/{title_id}/reviews/?ref_=tt_ururv_sm&spoilers=EXCLUDE", headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0'})
    html_page = urlopen(req).read()

    soup = BeautifulSoup(html_page, 'html.parser')

    ratings = soup.find_all('span', class_='ipc-rating-star--rating')
    titles = soup.find_all('h3', class_='ipc-title__text')
    reviews = soup.find_all('div', class_='ipc-html-content-inner-div')

    return [(f"{title.get_text()}\n{review.get_text(chr(10))}", rating.get_text()) for title, review, rating in zip(titles, reviews, ratings)]