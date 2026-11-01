from src.data_processor import load_csv, clean_dataframe
from src.sentiment_analyzer import analyze_dataframe

# 1. 读取原始评论
df = load_csv("data/raw/test_comments.csv")

# 2. 清洗评论
cleaned_df, stats = clean_dataframe(df)

# 3. 批量情感分析
result_df = analyze_dataframe(cleaned_df)

# 4. 输出分析结果
print("========== 批量情感分析结果 ==========")
print(result_df[["content", "sentiment_score", "sentiment_label"]])

# 5. 统计情感类别数量
print("\n========== 情感类别统计 ==========")
print(result_df["sentiment_label"].value_counts())

# 6. 统计各类别占比
print("\n========== 情感类别占比 ==========")
print(
    (result_df["sentiment_label"].value_counts(normalize=True) * 100)
    .round(2)
    .astype(str) + "%"
)

