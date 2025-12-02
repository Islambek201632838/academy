"""
ПОЛНЫЙ ГАЙД ПО FINE-TUNING ENCODERS И CLASSIFIERS
==================================================

Включает:
1. Fine-tune Sentence Transformers (для embeddings)
2. Fine-tune CrossEncoder (для ranking/classification)
3. Fine-tune BERT Classifier (для sequence classification)
4. Как использовать каждый тип модели
5. Сравнение производительности
"""

import json
import warnings

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample, losses
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

warnings.filterwarnings('ignore')


# ============================================================================
# ЧАСТЬ 1: FINE-TUNE SENTENCE TRANSFORMER (ДЛЯ EMBEDDING-BASED CLASSIFICATION)
# ============================================================================

def finetune_sentence_transformer():
    """Fine-tune Sentence Transformer для лучшего понимания категорий товаров"""

    print("\n" + "=" * 70)
    print("PART 1: FINE-TUNING SENTENCE TRANSFORMER")
    print("=" * 70)

    # Загрузить данные
    print("\n1. Loading data...")
    df = pd.read_csv('cross_encoded_dataset.csv')
    df = df.dropna(subset=['DESCRIPTION', 'Категория'])

    # Создать пары (товар, категория) для обучения
    print("2. Creating training data...")
    categories = sorted(df['Категория'].unique())

    train_examples = []
    for idx, row in df.iterrows():
        for category in categories:
            # Положительный пример - правильная категория
            if row['Категория'] == category:
                train_examples.append(
                    InputExample(
                        texts=[row['DESCRIPTION'], category],
                        label=1.0  # Сходство = 1 (правильная категория)
                    )
                )
            # Отрицательные примеры - неправильные категории
            else:
                if np.random.random() > 0.8:  # Сэмплируем 20% неправильных
                    train_examples.append(
                        InputExample(
                            texts=[row['DESCRIPTION'], category],
                            label=0.0  # Сходство = 0 (неправильная)
                        )
                    )

    print(f"Created {len(train_examples)} training examples")

    # Загрузить базовую модель
    print("\n3. Loading base model...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Создать DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

    # Создать loss функцию
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune
    print("\n4. Fine-tuning model...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=100,
        show_progress_bar=True
    )

    # Сохранить
    model_path = 'fine_tuned_sentence_transformer'
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")

    return model, categories


# ============================================================================
# ЧАСТЬ 2: FINE-TUNE CROSSENCODER (ДЛЯ RANKING/CLASSIFICATION)
# ============================================================================

def finetune_crossencoder():
    """Fine-tune CrossEncoder для классификации товаров"""

    print("\n" + "=" * 70)
    print("PART 2: FINE-TUNING CROSSENCODER")
    print("=" * 70)

    # Загрузить данные
    print("\n1. Loading data...")
    df = pd.read_csv('cross_encoded_dataset.csv')
    df = df.dropna(subset=['DESCRIPTION', 'Категория'])

    categories = sorted(df['Категория'].unique())
    category2id = {cat: idx for idx, cat in enumerate(categories)}

    # Создать пары (товар, категория) -> label
    print("2. Creating training data...")
    train_sentences = []
    train_labels = []

    for idx, row in df.iterrows():
        for category in categories:
            train_sentences.append([row['DESCRIPTION'], category])
            # Label = 1 если правильная категория, 0 иначе
            label = 1 if row['Категория'] == category else 0
            train_labels.append(label)

    print(f"Created {len(train_sentences)} sentence pairs")

    # Split на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        train_sentences, train_labels, test_size=0.2, random_state=42
    )

    # Загрузить базовую модель
    print("\n3. Loading base CrossEncoder...")
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Конвертировать в формат CrossEncoder
    train_samples = [
        InputExample(texts=[s[0], s[1]], label=l)
        for s, l in zip(X_train, y_train)
    ]

    val_samples = [
        InputExample(texts=[s[0], s[1]], label=l)
        for s, l in zip(X_val, y_val)
    ]

    # Fine-tune
    print("\n4. Fine-tuning CrossEncoder...")
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=100,
        show_progress_bar=True,
        evaluator=None  # Можешь добавить evaluator если нужно
    )

    # Сохранить
    model_path = 'fine_tuned_crossencoder'
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")

    # Сохранить mapping
    with open(f'{model_path}/category_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({
            'category2id': category2id,
            'id2category': {str(v): k for k, v in category2id.items()},
            'categories': categories
        }, f, ensure_ascii=False, indent=2)

    return model, categories


# ============================================================================
# ЧАСТЬ 3: FINE-TUNE BERT CLASSIFIER (SEQUENCE CLASSIFICATION)
# ============================================================================

def finetune_bert_classifier():
    """Fine-tune BERT для классификации товаров"""

    print("\n" + "=" * 70)
    print("PART 3: FINE-TUNING BERT CLASSIFIER")
    print("=" * 70)

    # Загрузить данные
    print("\n1. Loading data...")
    df = pd.read_csv('cross_encoded_dataset.csv')
    df = df.dropna(subset=['DESCRIPTION', 'Категория'])

    categories = sorted(df['Категория'].unique())
    category2id = {cat: idx for idx, cat in enumerate(categories)}
    id2category = {idx: cat for cat, idx in category2id.items()}

    df['label'] = df['Категория'].map(category2id)

    # Split
    print("\n2. Splitting data...")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Загрузить модель и токенайзер
    print("\n3. Loading model and tokenizer...")
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(categories),
        id2label=id2category,
        label2id=category2id
    )

    # Токенизировать
    print("\n4. Tokenizing data...")

    def tokenize_function(examples):
        return tokenizer(
            examples['DESCRIPTION'],
            padding='max_length',
            truncation=True,
            max_length=128
        )

    # Создать Dataset
    from datasets import Dataset

    train_dataset = Dataset.from_dict({
        'DESCRIPTION': train_df['DESCRIPTION'].tolist(),
        'label': train_df['label'].tolist()
    })

    val_dataset = Dataset.from_dict({
        'DESCRIPTION': val_df['DESCRIPTION'].tolist(),
        'label': val_df['label'].tolist()
    })

    test_dataset = Dataset.from_dict({
        'DESCRIPTION': test_df['DESCRIPTION'].tolist(),
        'label': test_df['label'].tolist()
    })

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    # Metrics
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    # Training arguments
    print("\n5. Setting training arguments...")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=2e-5,
    )

    # Trainer
    print("\n6. Training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate
    print("\n7. Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print("\nTest Results:")
    for key, value in test_results.items():
        if key.startswith('eval_'):
            print(f"  {key}: {value:.4f}")

    # Сохранить
    print("\n8. Saving model...")
    model_save_path = './fine_tuned_bert_classifier'
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    with open(f'{model_save_path}/category_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({
            'id2category': id2category,
            'category2id': category2id
        }, f, ensure_ascii=False, indent=2)

    print(f"✓ Model saved to: {model_save_path}")

    return model, tokenizer, id2category


# ============================================================================
# ЧАСТЬ 4: КАК ИСПОЛЬЗОВАТЬ FINE-TUNED МОДЕЛИ
# ============================================================================

def use_sentence_transformer_model():
    """Пример использования fine-tuned Sentence Transformer"""

    print("\n" + "=" * 70)
    print("USING FINE-TUNED SENTENCE TRANSFORMER")
    print("=" * 70)

    # Загрузить модель
    model = SentenceTransformer('fine_tuned_sentence_transformer')

    # Категории
    categories = [
        "Обувь", "Одежда", "Аксессуары и защита", "Сумки и чемоданы",
        "Спортивное оборудование", "Продукты питания и напитки",
        "Товары для дома и кухни", "Детские товары и игрушки",
        "Учебные и канцелярские товары", "Товары для творчества и искусства",
        "Строительные материалы", "Предметы быта и уборки", "Услуги"
    ]

    # Получить embeddings категорий
    category_embeddings = model.encode(categories)

    # Примеры товаров
    products = [
        "Кроссовки мужские Nike черные",
        "Футболка женская красная Adidas",
        "Рюкзак спортивный для школы",
        "Мороженое ванильное 500г",
    ]

    print("\nClassifications:")
    for product in products:
        # Получить embedding товара
        product_embedding = model.encode(product)

        # Найти самую похожую категорию
        similarities = model.util.pytorch_cos_sim(product_embedding, category_embeddings)[0]
        best_idx = np.argmax(similarities.cpu().numpy())

        print(f"\n  Product: {product}")
        print(f"  → Category: {categories[best_idx]}")
        print(f"  → Confidence: {similarities[best_idx]:.2%}")


def use_crossencoder_model():
    """Пример использования fine-tuned CrossEncoder"""

    print("\n" + "=" * 70)
    print("USING FINE-TUNED CROSSENCODER")
    print("=" * 70)

    # Загрузить модель
    model = CrossEncoder('fine_tuned_crossencoder')

    # Загрузить категории
    with open('fine_tuned_crossencoder/category_mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    categories = mapping['categories']

    # Примеры
    products = [
        "Кроссовки мужские Nike черные",
        "Футболка женская красная",
        "Рюкзак спортивный",
        "Мороженое ванильное 500г",
        "Услуга доставки товаров",
    ]

    print("\nClassifications:")
    for product in products:
        # Создать пары (товар, категория)
        pairs = [[product, cat] for cat in categories]

        # Получить scores
        scores = model.predict(pairs)
        best_idx = np.argmax(scores)

        print(f"\n  Product: {product}")
        print(f"  → Category: {categories[best_idx]}")
        print(f"  → Score: {scores[best_idx]:.4f}")


def use_bert_classifier_model():
    """Пример использования fine-tuned BERT Classifier"""

    print("\n" + "=" * 70)
    print("USING FINE-TUNED BERT CLASSIFIER")
    print("=" * 70)

    # Загрузить модель и токенайзер
    model_path = './fine_tuned_bert_classifier'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Загрузить mapping
    with open(f'{model_path}/category_mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    id2category = mapping['id2category']

    # Примеры
    products = [
        "Кроссовки мужские Nike черные",
        "Футболка женская красная",
        "Рюкзак спортивный",
        "Мороженое ванильное 500г",
        "Услуга доставки",
    ]

    print("\nClassifications:")
    for product in products:
        # Токенизировать
        inputs = tokenizer(
            product,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        # Предсказать
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=1).item()
        predicted_category = id2category[str(predicted_id)]
        confidence = torch.softmax(logits, dim=1).max().item()

        print(f"\n  Product: {product}")
        print(f"  → Category: {predicted_category}")
        print(f"  → Confidence: {confidence:.2%}")


# ============================================================================
# ЧАСТЬ 5: BATCH PREDICTION НА БОЛЬШОМ ДАТАСЕТЕ
# ============================================================================

def batch_predict_with_models(model_type='bert'):
    """Предсказание на всем датасете с использованием выбранной модели"""

    print("\n" + "=" * 70)
    print(f"BATCH PREDICTION WITH {model_type.upper()}")
    print("=" * 70)

    # Загрузить данные
    df = pd.read_csv('cross_encoded_dataset.csv')
    df = df.dropna(subset=['DESCRIPTION'])

    if model_type == 'bert':
        # Использовать BERT
        model_path = './fine_tuned_bert_classifier'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        with open(f'{model_path}/category_mapping.json', 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        id2category = mapping['id2category']

        def predict_fn(text):
            inputs = tokenizer(
                text, padding='max_length', truncation=True,
                max_length=128, return_tensors='pt'
            )
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_id = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
            return id2category[str(predicted_id)], confidence

    elif model_type == 'crossencoder':
        # Использовать CrossEncoder
        model = CrossEncoder('fine_tuned_crossencoder')

        with open('fine_tuned_crossencoder/category_mapping.json', 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        categories = mapping['categories']

        def predict_fn(text):
            pairs = [[text, cat] for cat in categories]
            scores = model.predict(pairs)
            best_idx = np.argmax(scores)
            return categories[best_idx], scores[best_idx]

    else:  # sentence_transformer
        # Использовать Sentence Transformer
        model = SentenceTransformer('fine_tuned_sentence_transformer')

        with open('fine_tuned_sentence_transformer/category_mapping.json', 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        categories = mapping['categories']

        category_embeddings = model.encode(categories)

        def predict_fn(text):
            text_embedding = model.encode(text)
            similarities = model.util.pytorch_cos_sim(text_embedding, category_embeddings)[0]
            best_idx = np.argmax(similarities.cpu().numpy())
            return categories[best_idx], similarities[best_idx].item()

    # Предсказать для всех товаров
    print("\nMaking predictions...")
    predictions = []
    confidences = []

    for idx, text in enumerate(df['DESCRIPTION']):
        if idx % 100 == 0:
            print(f"  Processed: {idx}/{len(df)}")
        pred, conf = predict_fn(text)
        predictions.append(pred)
        confidences.append(conf)

    df['predicted_category'] = predictions
    df['confidence'] = confidences

    # Сохранить
    output_file = f'batch_predictions_{model_type}.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✓ Predictions saved to: {output_file}")

    # Показать статистику
    print("\nPrediction Statistics:")
    print(df['predicted_category'].value_counts())
    print(f"\nAverage confidence: {df['confidence'].mean():.2%}")
    print(f"Min confidence: {df['confidence'].min():.2%}")
    print(f"Max confidence: {df['confidence'].max():.2%}")


# ============================================================================
# MAIN: ЗАПУСК ВСЕХ КОМПОНЕНТОВ
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMPLETE GUIDE: FINE-TUNING ENCODERS AND CLASSIFIERS")
    print("=" * 70)

    # Выбрать, что запустить
    import sys

    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        action = 'all'

    if action == 'all' or action == 'finetune':
        print("\n>>> Fine-tuning models...")

        # 1. Fine-tune Sentence Transformer
        print("\n[1/3] Sentence Transformer...")
        try:
            model_st, cats_st = finetune_sentence_transformer()
        except Exception as e:
            print(f"Error: {e}")

        # 2. Fine-tune CrossEncoder
        print("\n[2/3] CrossEncoder...")
        try:
            model_ce, cats_ce = finetune_crossencoder()
        except Exception as e:
            print(f"Error: {e}")

        # 3. Fine-tune BERT
        print("\n[3/3] BERT Classifier...")
        try:
            model_bert, tokenizer_bert, id2cat = finetune_bert_classifier()
        except Exception as e:
            print(f"Error: {e}")

    if action == 'all' or action == 'use':
        print("\n>>> Using fine-tuned models...")

        try:
            use_sentence_transformer_model()
        except Exception as e:
            print(f"Error with Sentence Transformer: {e}")

        try:
            use_crossencoder_model()
        except Exception as e:
            print(f"Error with CrossEncoder: {e}")

        try:
            use_bert_classifier_model()
        except Exception as e:
            print(f"Error with BERT: {e}")

    if action == 'all' or action == 'batch':
        print("\n>>> Batch predictions...")

        for model_type in ['bert', 'crossencoder', 'sentence_transformer']:
            try:
                batch_predict_with_models(model_type)
            except Exception as e:
                print(f"Error with {model_type}: {e}")

    print("\n" + "=" * 70)
    print("COMPLETED!")
    print("=" * 70)