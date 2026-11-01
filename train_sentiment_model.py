import sys

sys.stdout.reconfigure(encoding="utf-8")

from src.sentiment_model import train_sentiment_model


metrics = train_sentiment_model()

print("=== 情感分类模型训练完成 ===")

for key, value in metrics.items():
    print(f"{key}: {value}")
