import os
import pandas as pd
from lxml import etree

def extract_data_from_xml(xml_paths, output_csv):
    data = []

    for xml_path in xml_paths:
        with open(xml_path, "r", encoding="utf-8") as file:
            xml_data = file.read()
        tree = etree.fromstring(xml_data)

        for review in tree.xpath("//review"):
            review_text = review.xpath("text")[0].text
            aspects = review.xpath(".//aspect[@category='Whole']")

            for aspect in aspects:
                sentiment = aspect.get("sentiment")
                from_pos = int(aspect.get("from"))
                to_pos = int(aspect.get("to"))
                text_fragment = review_text[from_pos:to_pos]
                data.append((text_fragment, sentiment))

    df = pd.DataFrame(data, columns=["text", "sentiment"])
    label_mapping = {"positive": 1, "neutral": 0, "negative": -1}
    df["sentiment_label"] = df["sentiment"].map(label_mapping)

    initial_count = len(df)
    df = df.dropna(subset=["sentiment_label"])
    final_count = len(df)
    print(f"Удалено строк с неизвестными метками: {initial_count - final_count}")

    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория: {output_dir}")

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Данные успешно сохранены в {output_csv}")

if __name__ == "__main__":
    xml_paths = [
        "SentiRuEval_rest_markup_train.xml",
        "SentiRuEval_car_markup_train.xml"
    ]
    output_csv = "data/reviews.csv"
    extract_data_from_xml(xml_paths, output_csv)
