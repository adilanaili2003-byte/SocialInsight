from src.data_processor import load_csv, clean_dataframe, save_processed_data
from src.sentiment_analyzer import analyze_dataframe, save_sentiment_results
from src.topic_analyzer import (
    cluster_topics,
    get_topic_keywords,
    save_topic_results
)
from src.risk_analyzer import (
    analyze_risks,
    save_risk_comments,
    save_risk_summary
)
from src.report_generator import generate_report, save_report


def run_analysis(
    file,
    generate_ai_report=True,
    save_results=True,
    refine_sentiment=False
):
    """
    SocialInsight 统一舆情分析工作流。

    参数：
        file: CSV 文件路径或文件对象
        generate_ai_report: 是否调用大语言模型生成报告
        save_results: 是否保存分析结果
        refine_sentiment: 是否用大语言模型对不确定评论做情感精调

    返回：
        包含清洗统计、分析结果、主题关键词和报告的字典
    """

    # 1. 读取数据
    df = load_csv(file)

    # 2. 数据清洗
    cleaned_df, cleaning_stats = clean_dataframe(df)

    if cleaned_df.empty:
        raise ValueError("数据清洗后没有可分析的评论")

    # 3. 情感分析
    sentiment_df = analyze_dataframe(
        cleaned_df,
        refine_with_llm=refine_sentiment
    )

    # 4. 主题聚类
    topic_df, vectorizer, model = cluster_topics(
        sentiment_df,
        text_column="content",
        n_clusters=3
    )

    # 5. 提取主题关键词
    topic_keywords = get_topic_keywords(
        vectorizer,
        model,
        top_n=5
    )

    # 6. 舆情风险分析
    risk_result = analyze_risks(topic_df)

    # 7. 整理完整分析结果
    analysis_result = {
        "total_count": risk_result["total_count"],
        "negative_count": risk_result["negative_count"],
        "negative_ratio": risk_result["negative_ratio"],
        "risk_level": risk_result["risk_level"],
        "risk_counts": risk_result["risk_counts"],
        "topic_distribution": risk_result["topic_distribution"],
        "topic_keywords": topic_keywords
    }

    # 8. 保存分析结果
    if save_results:
        save_processed_data(cleaned_df)
        save_sentiment_results(sentiment_df)
        save_topic_results(topic_df)
        save_risk_comments(risk_result)
        save_risk_summary(risk_result)

    # 9. 大语言模型生成报告
    report = None

    if generate_ai_report:
        report = generate_report(analysis_result)

        if save_results:
            save_report(report)

    # 10. 返回完整结果
    return {
        "cleaning_stats": cleaning_stats,
        "cleaned_data": cleaned_df,
        "sentiment_data": sentiment_df,
        "topic_data": topic_df,
        "risk_result": risk_result,
        "topic_keywords": topic_keywords,
        "analysis_result": analysis_result,
        "report": report
    }
