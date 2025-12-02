# После обучения модели sklearn:
import joblib

clf = joblib.load('trained_classifier_sklearn.pkl')
vectorizer = joblib.load('trained_vectorizer_sklearn.pkl')

# Предсказание для нового товара
text = "Кроссовки мужские Nike"
X_new = vectorizer.transform([text])
prediction = clf.predict(X_new)[0]
confidence = clf.predict_proba(X_new)[0].max()

print(f"Категория: {prediction}")
print(f"Уверенность: {confidence:.2%}")