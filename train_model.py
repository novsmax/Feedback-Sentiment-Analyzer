import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_and_evaluate(csv_path, model_output=None):
    """
    Обучает модель для анализа тональности и оценивает её.

    Args:
        csv_path (str): Путь к CSV-файлу с данными.
        model_output (str): Путь для сохранения обученной модели (опционально).
    """
    # Загружаем данные
    df = pd.read_csv(csv_path)

    # Проверяем данные на наличие NaN
    if df["sentiment_label"].isnull().any():
        raise ValueError("В данных есть пропущенные значения в целевой переменной 'sentiment_label'.")

    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment_label"], test_size=0.2, random_state=42
    )

    # Преобразуем текст в TF-IDF векторы
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Обучаем модель
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Оцениваем модель
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    # Сохраняем модель, если указан путь
    if model_output:
        import pickle
        with open(model_output, "wb") as f:
            pickle.dump((model, vectorizer), f)
        print(f"Модель успешно сохранена в {model_output}")

def predict_text(model_path):
    """
    Предсказывает тональность текста с использованием сохранённой модели.

    Args:
        model_path (str): Путь к сохранённой модели.
    """
    import pickle
    # Загружаем модель
    with open(model_path, "rb") as f:
        model, vectorizer = pickle.load(f)

    # Словарь для преобразования предсказаний в текст
    sentiment_mapping = {
        1: "Положительный отзыв",
        0: "Нейтральный отзыв",
        -1: "Негативный отзыв"
    }

    print("\nВведите свои отзывы для проверки (по одному). Чтобы завершить, введите 'exit'.")
    while True:
        user_input = input("Введите отзыв: ")
        if user_input.lower() == "exit":
            print("Завершение работы.")
            break

        # Преобразуем текст в вектор
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]

        # Выводим результат
        print(f"Результат: {sentiment_mapping[prediction]}\n")

if __name__ == "__main__":
    csv_path = "data/reviews.csv"
    model_output = "data/sentiment_model.pkl"

    # Обучение модели
    train_and_evaluate(csv_path, model_output)

    # Предсказания для пользовательских данных
    predict_text(model_output)
