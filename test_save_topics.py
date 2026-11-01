from src.data_processor import load_csv, clean_dataframe
from src.topic_analyzer import (
    cluster_topics,
    save_topic_results
)
import pandas as pd
import os

# 1. 读取原始数据
df = load_csv("data/raw/test_comments.csv")

# 2. 清洗数据
cleaned_df, stats = clean_dataframe(df)

# 3. 执行主题聚类
result_df, vectorizer, model = cluster_topics(
    cleaned_df,
    text_column="content",
    n_clusters=3
)

# 4. 保存主题分析结果
output_path = save_topic_results(result_df)

# 5. 重新读取保存的文件
saved_df = pd.read_csv(output_path, encoding="utf-8-sig")

# 6. 检查保存结果
print("\n========== 主题分析保存测试 ==========")
print("文件是否存在：", os.path.exists(output_path))
print("保存前数据条数：", len(result_df))
print("保存后数据条数：", len(saved_df))
print("是否包含 topic_id：", "topic_id" in saved_df.columns)

print("\n保存后的主题分布：")
print(saved_df["topic_id"].value_counts().sort_index())

# 7. 验证数据
assert len(result_df) == len(saved_df)
assert "topic_id" in saved_df.columns
assert list(result_df["topic_id"]) == list(saved_df["topic_id"])

print("\n测试通过：主题分析结果保存成功！")

