from pathlib import Path

import pandas as pd

from src.data_processor import load_csv, clean_dataframe
from src.sentiment_analyzer import analyze_dataframe
from src.topic_analyzer import cluster_topics, get_topic_keywords
from src.risk_analyzer import (
    analyze_risks,
    save_risk_comments,
    save_risk_summary
)


# 1. 读取原始数据
print("\n========== 第一步：读取原始数据 ==========")

df = load_csv("data/raw/test_comments.csv")

print(f"原始评论数量：{len(df)}")


# 2. 数据清洗
print("\n========== 第二步：数据清洗 ==========")

cleaned_df, stats = clean_dataframe(df)

print(f"清洗后评论数量：{len(cleaned_df)}")
print(f"删除空评论：{stats['removed_empty']}")
print(f"删除重复评论：{stats['removed_duplicates']}")


# 3. 情感分析
print("\n========== 第三步：情感分析 ==========")

sentiment_df = analyze_dataframe(cleaned_df)

print(
    sentiment_df[
        ["content", "sentiment_score", "sentiment_label"]
    ]
)


# 4. 主题聚类
print("\n========== 第四步：主题分析 ==========")

result_df, vectorizer, model = cluster_topics(
    sentiment_df,
    text_column="content",
    n_clusters=3
)

topic_keywords = get_topic_keywords(
    vectorizer,
    model,
    top_n=5
)

print("\n主题关键词：")

for topic_id, keywords in topic_keywords.items():
    print(f"主题 {topic_id}：{keywords}")


# 5. 舆情风险分析
print("\n========== 第五步：舆情风险分析 ==========")

risk_result = analyze_risks(result_df)

print(f"评论总数：{risk_result['total_count']}")
print(f"负面评论数量：{risk_result['negative_count']}")
print(f"负面评论比例：{risk_result['negative_ratio']:.2%}")
print(f"风险等级：{risk_result['risk_level']}")

print("\n风险类别统计：")

for category, count in risk_result["risk_counts"].items():
    print(f"{category}：{count}")

print("\n负面评论主题分布：")

for topic_id, count in risk_result["topic_distribution"].items():
    print(f"主题 {topic_id}：{count} 条")


# 6. 保存完整分析结果
print("\n========== 第六步：保存分析结果 ==========")

output_path = "data/processed/full_analysis_results.csv"

Path(output_path).parent.mkdir(
    parents=True,
    exist_ok=True
)

result_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

# 保存风险评论明细
risk_comments_path = save_risk_comments(risk_result)

# 保存风险统计汇总
risk_summary_path = save_risk_summary(risk_result)


# 7. 检查保存结果
print("\n========== 第七步：检查保存结果 ==========")

saved_df = pd.read_csv(
    output_path,
    encoding="utf-8-sig"
)

print(f"完整分析结果：{output_path}")
print(f"风险评论明细：{risk_comments_path}")
print(f"风险统计汇总：{risk_summary_path}")

print(f"\n最终数据条数：{len(saved_df)}")

print("\n最终数据字段：")
print(saved_df.columns.tolist())


# 8. 验证
assert len(saved_df) == len(cleaned_df)

assert "sentiment_score" in saved_df.columns
assert "sentiment_label" in saved_df.columns
assert "topic_id" in saved_df.columns

assert Path(risk_comments_path).exists()
assert Path(risk_summary_path).exists()

print("\n整体 Pipeline 测试通过！")
print("数据清洗、情感分析、主题分析和舆情风险分析已经成功串联。")