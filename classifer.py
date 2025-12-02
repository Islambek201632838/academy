import pandas as pd
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Загрузить классификатор
print("Loading classifier model...")
classifier = pipeline("zero-shot-classification", model="xlm-roberta-large")

# Ваши категории
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

# Функция классификации
def classify_product(product_text):
    """Классифицирует продукт в одну из категорий"""
    try:
        result = classifier(product_text, categories)
        return result['labels'][0]  # Возвращаем категорию с highest score
    except:
        return "Неизвестно"

# Читаем CSV файл
print("Reading CSV file...")
with open('esf_fulll_202511211949.csv', 'r', encoding='utf-8') as f:
    lines = [line.strip().replace('"', '')
             for line in f.readlines()
             if line.strip()]

lines = lines[1:]
df = pd.DataFrame({'DESCRIPTION': lines})

print(f"Total products: {len(df)}")
print(f"First few rows:\n{df.head()}\n")

# Добавляем колонку с категориями
print("Classifying products... (this may take a few minutes)")
df['Категория'] = df['DESCRIPTION'].apply(classify_product)

# Показываем результаты
print("\nClassification completed!")
print(df.head(20))

# Сохраняем результат
output_path = 'classifed_dataset.csv'
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\nSaved to: {output_path}")

# Статистика
print("\nCategory distribution:")
print(df['Категория'].value_counts())