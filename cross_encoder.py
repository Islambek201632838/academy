import pandas as pd
from sentence_transformers import CrossEncoder
import warnings

warnings.filterwarnings('ignore')

print("Loading model...")
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

categories = [
    "Обувь",
    "Одежда",
    "Аксессуары и защита",
    "Сумки и чемоданы",
    "Спортивное оборудование",
    "Продукты питания и напитки",
    "Товары для дома и кухни",
    "Детские товары и игрушки",
    "Учебные и канцелярские товары",
    "Товары для творчества и искусства",
    "Строительные материалы",
    "Предметы быта и уборки",
    "Услуги"
]


def classify_product(product_text):
    """Классифицирует продукт в одну из категорий"""
    try:
        category_pairs = [[product_text, cat] for cat in categories]
        scores = model.predict(category_pairs)

        best_idx = scores.argmax()
        return categories[best_idx]
    except:
        return "Неизвестно"


print("Reading CSV file...")
with open('esf_fulll_202511211949.csv', 'r', encoding='utf-8') as f:
    lines = [line.strip().replace('"', '')
             for line in f.readlines()
             if line.strip()]

lines = lines[1:]
df = pd.DataFrame({'DESCRIPTION': lines})

print(f"Total products: {len(df)}")
print(f"First few rows:\n{df.head()}\n")

print("Classifying products... (this may take a few minutes)")
df['Категория'] = df['DESCRIPTION'].apply(classify_product)

print("\nClassification completed!")
print(df.head(20))

output_path = 'cross_encoded_dataset.csv'
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\nSaved to: {output_path}")

print("\nCategory distribution:")
print(df['Категория'].value_counts())