import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
import spacy_udpipe

nlp = spacy_udpipe.load("ru")

# Расширенный словарь с эмоциональными словами
emotion_dict = {
    "positive": [
        "отличный", "прекрасный", "удобный", "лучший", "супер", "великолепный", "замечательный", "классный", "восторг",
        "позитивный",
        "радостный", "вдохновляющий", "удивительный", "чудесный", "отлично", "хороший", "потрясающий", "фантастический"
    ],
    "negative": [
        "ужасный", "плохой", "разочарован", "не рекомендую", "проблемы", "не советую", "ужас", "отвратительный",
        "кошмар",
        "катастрофа", "неудачный", "несчастный", "грустный", "неприятный", "мучительный", "провал", "страшный",
        "неудача"
    ],
    "neutral": [
        "нормальный", "обычный", "средний", "стандартный", "нейтральный", "проходной", "так себе", "неплохой",
        "простой", "обыденный",
        "среднестатистический", "невыразительный", "посредственный", "безразличный", "функциональный"
    ]
}


def classify_by_rules(text, emotion_dict):
    # Лемматизируем текст с помощью spaCy
    doc = nlp(text.lower())  # Приводим текст к нижнему регистру
    lemmatized_text = [token.lemma_ for token in doc]  # Лемматизация слов в тексте

    matches = {
        "positive": 0,
        "negative": 0,
        "neutral": 0
    }

    # Проверка наличия хотя бы нескольких лемм из каждой категории
    for sentiment, words in emotion_dict.items():
        # Лемматизируем слова из словаря эмоций
        lemmatized_words = set([nlp(word)[0].lemma_ for word in words])

        # Считаем совпадения лемм
        matches[sentiment] = sum(1 for word in lemmatized_text if word in lemmatized_words)

    # Возвращаем категорию с наибольшим количеством совпадений
    if matches["positive"] > 0 and (
            matches["positive"] >= matches["negative"] or matches["positive"] > matches["neutral"]):
        return "positive"
    elif matches["negative"] > 0 and (
            matches["negative"] >= matches["positive"] or matches["negative"] > matches["neutral"]):
        return "negative"
    elif matches["neutral"] > 0:
        return "neutral"

    return None


def hybrid_classification(text, model, vectorizer, emotion_dict):

    rule_based_result = classify_by_rules(text, emotion_dict)
    if rule_based_result:
        print("Результат классификации по правилам:", rule_based_result)
        return rule_based_result

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    sentiment_mapping = {
        1: "positive",
        0: "neutral",
        -1: "negative"
    }
    return sentiment_mapping[prediction]

def train_and_save_models(csv_path, model_outputs):
    df = pd.read_csv(csv_path)
    if df["sentiment_label"].isnull().any():
        raise ValueError("В данных есть пропущенные значения в целевой переменной 'sentiment_label'.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment_label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Логистическая регрессия": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Наивный Байес": MultinomialNB(),
        "Дерево решений": DecisionTreeClassifier(),
        "Метод опорных векторов": SVC(kernel="linear", class_weight="balanced")
    }

    for model_name, model in models.items():
        # print(f"\n--- Обучение модели: {model_name} ---")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        # print(f"Метрики для модели {model_name}:")
        # print(classification_report(y_test, y_pred))

        model_path = model_outputs.get(model_name)
        if model_path:
            with open(model_path, "wb") as f:
                pickle.dump((model, vectorizer), f)
            print(f"Модель {model_name} сохранена в {model_path}")

def hybrid_predict_all(model_paths, text):

    print(f"\nАнализ текста: {text}\n")
    for model_name, model_path in model_paths.items():
        with open(model_path, "rb") as f:
            model, vectorizer = pickle.load(f)

        result = hybrid_classification(text, model, vectorizer, emotion_dict)
        print(f"Модель: {model_name}")
        print(f"Эмоциональная окраска: {result}\n")

if __name__ == "__main__":
    csv_path = "data/reviews.csv"
    model_outputs = {
        "Логистическая регрессия": "data/logistic_model.pkl",
        "Наивный Байес": "data/naive_bayes_model.pkl",
        "Дерево решений": "data/decision_tree_model.pkl",
        "Метод опорных векторов": "data/svm_model.pkl"
    }

    train_and_save_models(csv_path, model_outputs)

    print("\nВведите отзывы для анализа (введите 'exit' для завершения):")
    while True:
        user_input = input("Отзыв: ")
        if user_input.lower() == "exit":
            break
        hybrid_predict_all(model_outputs, user_input)
