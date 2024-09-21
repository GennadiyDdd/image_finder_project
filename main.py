import os
import sys
import requests
from langchain_openai import OpenAI  # Обновленный импорт
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time
from requests.exceptions import RequestException

# Загрузка переменных окружения из файла .env
load_dotenv()

# Получение переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEARCH_ENGINE = os.getenv("SEARCH_ENGINE")  # 'google' или 'duckduckgo'
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def make_request_with_retries(url, params, max_retries=5):
    """Выполняет HTTP-запрос с повторными попытками при ошибках."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            if response.status_code == 429:
                print("Превышен лимит запросов. Пытаюсь снова...")
                time.sleep(2 ** attempt)  # Экспоненциальная задержка
                continue
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"Ошибка запроса: {e}. Пытаюсь снова...")
            time.sleep(2 ** attempt)
    print("Не удалось выполнить запрос после нескольких попыток.")
    return None

def extract_keywords(text):
    """Извлекает ключевые слова из текста новости с помощью LLM."""
    if not OPENAI_API_KEY:
        print("Ошибка: отсутствует OPENAI_API_KEY.")
        sys.exit(1)
    llm = OpenAI(api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Извлеките ключевые слова или фразы из следующего текста для поиска изображений:\n\n{text}"
    )
    keywords = llm.invoke(prompt.format(text=text))  # Используем invoke
    return keywords.strip()

def search_images_google(query):
    """Ищет изображения через Google Custom Search API с повторными попытками."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "cx": GOOGLE_CSE_ID,
        "key": GOOGLE_API_KEY,
        "searchType": "image",
        "num": 5
    }
    data = make_request_with_retries(url, params)
    if data:
        return data.get('items', [])
    return []

def search_images_duckduckgo(query):
    """Ищет изображения через SerpAPI с использованием DuckDuckGo с повторными попытками."""
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "engine": "duckduckgo",
        "api_key": SERPAPI_API_KEY,
        "no_html": "1",
        "output": "json"
    }
    data = make_request_with_retries(url, params)
    if data:
        return data.get('images_results', [])
    return []

def evaluate_relevance(images, text):
    """Оценивает релевантность изображений к тексту новости с помощью LLM."""
    if not OPENAI_API_KEY:
        print("Ошибка: отсутствует OPENAI_API_KEY.")
        sys.exit(1)
    llm = OpenAI(api_key=OPENAI_API_KEY)
    best_image = None
    best_score = float('-inf')
    for image in images:
        image_url = image.get('link') or image.get('thumbnail')
        if not image_url:
            continue
        prompt = PromptTemplate(
            input_variables=["text", "image_url"],
            template=(
                "Оцените релевантность следующего изображения к данному тексту по шкале от 1 до 10.\n\n"
                "Текст новости:\n{text}\n\nURL изображения: {image_url}\n\nОценка:"
            )
        )
        score_text = llm.invoke(prompt.format(text=text, image_url=image_url))  # Используем invoke
        try:
            score = float(score_text.strip())
        except ValueError:
            score = 0
        if score > best_score:
            best_score = score
            best_image = {
                'url': image_url,
                'description': image.get('title', 'Нет описания')
            }
    return best_image

def main():
    """Основная функция приложения."""
    # Проверка наличия ключей API
    if not OPENAI_API_KEY:
        print("Ошибка: отсутствует OPENAI_API_KEY.")
        sys.exit(1)
    if SEARCH_ENGINE == "google":
        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            print("Ошибка: отсутствует GOOGLE_API_KEY или GOOGLE_CSE_ID.")
            sys.exit(1)
    elif SEARCH_ENGINE == "duckduckgo":
        if not SERPAPI_API_KEY:
            print("Ошибка: отсутствует SERPAPI_API_KEY для DuckDuckGo.")
            sys.exit(1)
    else:
        print("Ошибка: неверное значение SEARCH_ENGINE. Используйте 'google' или 'duckduckgo'.")
        sys.exit(1)

    # Ввод текста новости
    text = input("Введите текст новости: ")

    # Извлечение ключевых слов
    print("Извлечение ключевых слов...")
    keywords = extract_keywords(text)
    print(f"Ключевые слова: {keywords}")

    # Поиск изображений
    print(f"Поиск изображений через {SEARCH_ENGINE}...")
    if SEARCH_ENGINE == "google":
        images = search_images_google(keywords)
    elif SEARCH_ENGINE == "duckduckgo":
        images = search_images_duckduckgo(keywords)
    else:
        print("Неверная поисковая система.")
        return

    if not images:
        print("Изображения не найдены.")
        return

    # Оценка релевантности
    print("Оценка релевантности изображений...")
    best_image = evaluate_relevance(images, text)

    if best_image:
        print("URL выбранного изображения:", best_image['url'])
        print("Описание изображения:", best_image['description'])
    else:
        print("Не удалось выбрать релевантное изображение.")

if __name__ == "__main__":
    main()
