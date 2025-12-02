import warnings

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

print("=" * 60)
print("TRAINING CLASSIFICATION MODEL (SKLEARN VERSION)")
print("=" * 60)

# 1. ЗАГРУЗИТЬ ДАННЫЕ
print("\n1. Loading data...")
df = pd.read_csv('cross_encoded_dataset.csv')

# Удалить пустые значения
df = df.dropna(subset=['DESCRIPTION', 'Категория'])
print(f"Total samples: {len(df)}")
print(f"\nCategory distribution:")
print(df['Категория'].value_counts())

# 2. SPLIT НА TRAIN/TEST
print("\n2. Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    df['DESCRIPTION'],
    df['Категория'],
    test_size=0.2,
    random_state=42,
    stratify=df['Категория']
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# 3. ВЕКТОРИЗИРОВАТЬ ТЕКСТ
print("\n3. Vectorizing text (TF-IDF)...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    lowercase=True,
    stop_words=None
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# 4. ОБУЧИТЬ МОДЕЛЬ
print("\n4. Training Logistic Regression model...")
clf = LogisticRegression(
    max_iter=1000,
    random_state=42,
    multi_class='multinomial',
    solver='lbfgs',
    n_jobs=-1
)

clf.fit(X_train_tfidf, y_train)
print("✓ Model trained")

# 5. ОЦЕНИТЬ МОДЕЛЬ
print("\n5. Evaluating model...")
y_pred = clf.predict(X_test_tfidf)
y_pred_proba = clf.predict_proba(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. СОХРАНИТЬ МОДЕЛЬ
print("\n6. Saving model...")
model_path = 'trained_classifier_sklearn.pkl'
vectorizer_path = 'trained_vectorizer_sklearn.pkl'

joblib.dump(clf, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"Model saved: {model_path}")
print(f"Vectorizer saved: {vectorizer_path}")

# 7. ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ
print("\n7. Creating prediction function...")


def predict_category(text):
    """Предсказать категорию для нового текста"""
    X_new = vectorizer.transform([text])
    prediction = clf.predict(X_new)[0]
    probabilities = clf.predict_proba(X_new)[0]
    confidence = probabilities.max()

    return prediction, confidence


# 8. ТЕСТИРИРОВАНИЕ
print("\n8. Testing predictions on examples...")
test_examples = [
    "Кроссовки мужские Nike черные",
    "Футболка женская красная Adidas",
    "Рюкзак спортивный для школы",
    "Мороженое ванильное 500г",
    "Услуга доставки товаров",
    "Гуашь краска для рисования",
    "Шапка зимняя теплая",
]

print("\nPredictions:")
for text in test_examples:
    category, confidence = predict_category(text)
    print(f"  '{text}'")
    print(f"    → {category} ({confidence:.2%})\n")

# 9. ОЦЕНКА НА ВСЕМ ДАТАСЕТЕ
print("\n9. Full dataset predictions...")
df['predicted_category'] = df['DESCRIPTION'].apply(lambda x: predict_category(x)[0])
df['confidence'] = df['DESCRIPTION'].apply(lambda x: predict_category(x)[1])

accuracy_full = (df['Категория'] == df['predicted_category']).sum() / len(df)
print(f"Overall accuracy on full dataset: {accuracy_full:.2%}")

# Матрица ошибок
print("\nConfusion Matrix:")
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(df['Категория'], df['predicted_category'])
print(f"Shape: {cm.shape}")

# 10. СОХРАНИТЬ РЕЗУЛЬТАТЫ
print("\n10. Saving results...")
df_results = df[['DESCRIPTION', 'Категория', 'predicted_category', 'confidence']].copy()
df_results.to_csv('predictions_sklearn.csv', index=False, encoding='utf-8')
print(f"Predictions saved: predictions_sklearn.csv")

# Статистика ошибок
print("\n11. Error Analysis...")
errors = df[df['Категория'] != df['predicted_category']]
print(f"Total errors: {len(errors)} ({len(errors) / len(df) * 100:.2f}%)")
print("\nErrors by category:")
print(errors['Категория'].value_counts().head(10))

print("\n" + "=" * 60)
print("TRAINING COMPLETED!")
print("=" * 60)
print(f"\nFiles created:")
print(f"  1. {model_path} - обученная модель")
print(f"  2. {vectorizer_path} - токенизатор")
print(f"  3. predictions_sklearn.csv - предсказания")