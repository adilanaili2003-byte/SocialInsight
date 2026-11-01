import pandas as pd
from src.risk_analyzer import analyze_risks

print("========== 风险分析异常输入测试 ==========")

# 测试1：空 DataFrame
print("\n测试1：空 DataFrame")

try:
    df_empty = pd.DataFrame()
    analyze_risks(df_empty)
    print("未触发异常，需要检查代码")
except Exception as e:
    print(f"成功捕获异常：{type(e).__name__}: {e}")


# 测试2：缺少 content 字段
print("\n测试2：缺少 content 字段")

try:
    df_missing_content = pd.DataFrame({
        "sentiment_label": ["负面"],
        "topic_id": [0]
    })
    analyze_risks(df_missing_content)
    print("未触发异常，需要检查代码")
except Exception as e:
    print(f"成功捕获异常：{type(e).__name__}: {e}")


# 测试3：缺少 sentiment_label 字段
print("\n测试3：缺少 sentiment_label 字段")

try:
    df_missing_sentiment = pd.DataFrame({
        "content": ["产品质量太差了"],
        "topic_id": [0]
    })
    analyze_risks(df_missing_sentiment)
    print("未触发异常，需要检查代码")
except Exception as e:
    print(f"成功捕获异常：{type(e).__name__}: {e}")


# 测试4：缺少 topic_id 字段
print("\n测试4：缺少 topic_id 字段")

try:
    df_missing_topic = pd.DataFrame({
        "content": ["产品质量太差了"],
        "sentiment_label": ["负面"]
    })
    analyze_risks(df_missing_topic)
    print("未触发异常，需要检查代码")
except Exception as e:
    print(f"成功捕获异常：{type(e).__name__}: {e}")

