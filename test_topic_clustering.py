from src.data_processor import load_csv, clean_dataframe
from src.topic_analyzer import cluster_topics, get_topic_keywords

# 1. 读取数据
df = load_csv("data/raw/test_comments.csv")

# 2. 清洗数据
cleaned_df, stats = clean_dataframe(df)

# 3. 执行主题聚类
result_df, vectorizer, model = cluster_topics(
    cleaned_df,
    text_column="content",
    n_clusters=3
)

# 4. 提取每个主题的关键词
topic_keywords = get_topic_keywords(
    vectorizer,
    model,
    top_n=5
)

# 5. 输出聚类结果
print("========== KMeans 主题聚类结果 ==========")

for _, row in result_df.iterrows():
    print(f"\n评论：{row['content']}")
    print(f"所属主题：主题 {row['topic_id']}")

# 6. 输出主题关键词
print("\n========== 各主题关键词 ==========")

for topic_id, keywords in topic_keywords.items():
    print(f"主题 {topic_id}：{keywords}")

# 7. 输出主题数量统计
print("\n========== 主题评论数量 ==========")
print(result_df["topic_id"].value_counts().sort_index())

