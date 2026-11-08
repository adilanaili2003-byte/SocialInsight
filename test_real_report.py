from src.data_processor import load_csv, clean_dataframe
from src.sentiment_analyzer import analyze_dataframe
from src.topic_analyzer import cluster_topics, get_topic_keywords
from src.risk_analyzer import analyze_risks
from src.report_generator import generate_report, save_report


# 1. 读取并清洗数据
print("\n========== 读取数据 ==========")

df = load_csv("data/raw/test_comments.csv")
cleaned_df, stats = clean_dataframe(df)

print(f"原始评论：{stats['original_count']}")
print(f"清洗后评论：{stats['final_count']}")


# 2. 情感分析
print("\n========== 情感分析 ==========")

sentiment_df = analyze_dataframe(cleaned_df)


# 3. 主题分析
print("\n========== 主题分析 ==========")

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


# 4. 风险分析
print("\n========== 风险分析 ==========")

risk_result = analyze_risks(result_df)


# 5. 整理报告数据
sentiment_counts = sentiment_df["sentiment_label"].value_counts()

analysis_result = {
    "total_count": risk_result["total_count"],
    "positive_count": int(sentiment_counts.get("正面", 0)),
    "neutral_count": int(sentiment_counts.get("中性", 0)),
    "negative_count": risk_result["negative_count"],
    "negative_ratio": risk_result["negative_ratio"],
    "risk_level": risk_result["risk_level"],
    "risk_counts": risk_result["risk_counts"],
    "topic_distribution": risk_result["topic_distribution"],
    "topic_keywords": topic_keywords
}


# 6. 调用 DeepSeek 生成报告
print("\n========== 调用 DeepSeek-V4-Pro ==========")

report = generate_report(analysis_result)

print("\n报告生成成功！")
print("\n========== 报告内容 ==========\n")
print(report)


# 7. 保存报告
print("\n========== 保存报告 ==========")

output_path = save_report(report)

print(f"报告保存位置：{output_path}")

print("\n真实 API 报告生成测试完成！")