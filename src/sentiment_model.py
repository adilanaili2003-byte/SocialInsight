import pickle
from pathlib import Path

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "sentiment_model.pkl"

_model_cache = {}


def tokenize_for_sentiment(text):
    """用 jieba 分词后返回空格分隔的文本，供 TF-IDF 使用。"""

    text = str(text).strip()
    tokens = [
        token
        for token in jieba.cut(text)
        if token.strip() and token.isalnum()
    ]
    return " ".join(tokens)


def build_pipeline():
    """构建 TF-IDF + 逻辑回归的情感分类流水线。"""

    return Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    token_pattern=r"\S+",
                    ngram_range=(1, 2),
                    max_features=50000,
                    min_df=2,
                    sublinear_tf=True
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )


def train_sentiment_model(
    data_path=None,
    model_path=None,
    test_size=0.2,
    random_state=42
):
    """在带 label 标注的数据上训练模型，并返回留出集的评估指标。"""

    data_path = (
        Path(data_path)
        if data_path
        else PROJECT_ROOT / "data" / "raw" / "waimai_10k.csv"
    )
    model_path = (
        Path(model_path)
        if model_path
        else DEFAULT_MODEL_PATH
    )

    df = pd.read_csv(data_path, encoding="utf-8-sig")

    for column in ["content", "label"]:
        if column not in df.columns:
            raise ValueError(f"训练数据缺少必要字段：{column}")

    df = df.dropna(subset=["content", "label"]).copy()
    df["tokens"] = df["content"].apply(tokenize_for_sentiment)

    X = df["tokens"]
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    positive_precision, positive_recall, positive_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=[1],
            zero_division=0
        )
    )

    negative_precision, negative_recall, negative_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=[0],
            zero_division=0
        )
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as file:
        pickle.dump(pipeline, file)

    return {
        "model_path": str(model_path),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": round(accuracy, 4),
        "positive_precision": round(positive_precision[0], 4),
        "positive_recall": round(positive_recall[0], 4),
        "positive_f1": round(positive_f1[0], 4),
        "negative_precision": round(negative_precision[0], 4),
        "negative_recall": round(negative_recall[0], 4),
        "negative_f1": round(negative_f1[0], 4)
    }


def load_sentiment_model(model_path=None):
    """加载已训练的情感模型，文件不存在时返回 None。"""

    model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not model_path.exists():
        return None

    if model_path not in _model_cache:
        with open(model_path, "rb") as file:
            _model_cache[model_path] = pickle.load(file)

    return _model_cache[model_path]


def predict_sentiment(model, text):
    """返回模型对文本判断为正面（label=1）的概率。"""

    tokens = tokenize_for_sentiment(text)

    if not tokens:
        return None

    classifier = model.named_steps["classifier"]
    positive_index = list(classifier.classes_).index(1)
    probability = model.predict_proba([tokens])[0]

    return float(probability[positive_index])
