import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

nltk.download('punkt')
from nltk.tokenize import word_tokenize

def get_game_id(game_name):
    search_url = f"https://store.steampowered.com/search/?term={game_name}"
    response = requests.get(search_url)
    if response.status_code != 200:
        print("Error: Steam 검색 페이지 호출 실패")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    game_link = soup.find("a", {"class": "search_result_row"})
    if game_link and "app" in game_link['href']:
        game_id = game_link['href'].split("/app/")[1].split("/")[0]
        return int(game_id)
    else:
        print("게임을 찾을 수 없습니다.")
        return None

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def fetch_negative_reviews(game_name, num_reviews=500):
    game_id = get_game_id(game_name)
    if not game_id:
        print("게임 ID를 찾을 수 없습니다.")
        return []
    
    print(f"게임 '{game_name}'의 ID는 {game_id}입니다.")
    STEAM_API_URL = f"https://store.steampowered.com/appreviews/{game_id}"
    all_reviews = []
    params = {
        'json': '1',
        'filter': 'recent',
        'language': 'english',
        'review_type': 'negative',
        'purchase_type': 'all',
        'num_per_page': 100,
        'cursor': '*'
    }
    
    while len(all_reviews) < num_reviews:
        response = requests.get(STEAM_API_URL, params=params)
        if response.status_code != 200:
            print("Error: API 호출 실패")
            break

        data = response.json()
        reviews = data.get('reviews', [])
        all_reviews.extend(reviews)

        if 'cursor' in data:
            params['cursor'] = data['cursor']
        else:
            break
        
        time.sleep(1)
        
        if len(reviews) == 0:
            break
    
    review_data = []
    for review in all_reviews[:num_reviews]:
        processed_review = preprocess_text(review['review'])
        review_info = {
            'review': review['review'],
            'processed_review': processed_review,
            'playtime_forever': review['author']['playtime_forever'] / 60,
            'timestamp_created': review['timestamp_created']
        }
        review_data.append(review_info)

    return review_data

def extract_keywords_tfidf(reviews, top_n=5):
    processed_reviews = [review['processed_review'] for review in reviews]
    vectorizer = TfidfVectorizer(max_df=0.85, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(processed_reviews)
    feature_names = vectorizer.get_feature_names_out()

    keywords = []
    for row in tfidf_matrix:
        sorted_indices = row.toarray().flatten().argsort()[::-1]
        top_keywords = [feature_names[i] for i in sorted_indices[:top_n]]
        keywords.append(top_keywords)
    
    return keywords

# 예시 사용
game_name = "Elden Ring"  # 예시 게임 이름
reviews = fetch_negative_reviews(game_name, num_reviews=500)

# 키워드 추출
keywords = extract_keywords_tfidf(reviews, top_n=10)

# DataFrame에 키워드 추가 후 CSV로 저장
df = pd.DataFrame(reviews)
df['keywords'] = keywords
df.to_csv("reviews_with_keywords.csv", index=False, encoding='utf-8-sig')

print("reviews_with_keywords.csv 파일로 저장이 완료되었습니다.")




# 각 리뷰의 키워드를 한 리스트로 합치기
all_keywords = [keyword for sublist in keywords for keyword in sublist]

# 키워드 빈도 계산 및 정렬
keyword_counts = Counter(all_keywords)
sorted_keywords = keyword_counts.most_common()  # 빈도 순으로 정렬

# DataFrame으로 변환하여 보기 좋게 출력
sorted_keywords_df = pd.DataFrame(sorted_keywords, columns=['Keyword', 'Frequency'])
sorted_keywords_df.to_csv("sorted_keywords.csv", index=False, encoding='utf-8-sig')

print("자주 등장하는 키워드 순으로 정렬된 결과를 sorted_keywords.csv 파일로 저장했습니다.")
print(sorted_keywords_df.head(10))  # 상위 10개 출력