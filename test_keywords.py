from src.data_processor import load_csv, clean_dataframe
from src.topic_analyzer import extract_keywords

# 1. 读取评论数据
df = load_csv("data/raw/test_comments.csv")

# 2. 清洗数据
cleaned_df, stats = clean_dataframe(df)

# 3. 提取高频关键词
keywords = extract_keywords(
    cleaned_df,
    text_column="content",
    top_n=10
)

# 4. 输出结果
print("========== 高频关键词提取测试 ==========")
print(f"参与分析的评论数量：{len(cleaned_df)}")

print("\n========== 高频关键词及出现次数 ==========")

for word, count in keywords:
    print(f"{word}: {count}")

