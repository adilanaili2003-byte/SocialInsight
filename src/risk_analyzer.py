import pandas as pd
import json
from pathlib import Path

# 风险等级阈值（按负面评论占比）
# 说明：当前风险等级仅由负面评论占比决定，是项目初期的简化规则。
# 注意「风险 ≠ 负面情感」：舆情风险还应综合考虑命中的风险类别及其严重程度
# （例如「过敏 / 被骗」比「价格贵」更需要关注）、负面情绪强度、评论影响力等因素。
# 后续可在风险等级判断中引入风险类别权重与情感强度，使结果更贴合真实舆情。
HIGH_RISK_THRESHOLD = 0.5
MEDIUM_RISK_THRESHOLD = 0.2


# 舆情风险关键词词典
RISK_KEYWORDS = {
    "产品质量": [
        "质量差",
        "质量问题",
        "质量太差",
        "损坏",
        "故障",
        "坏了"
    ],

    "物流服务": [
        "物流慢",
        "物流太慢",
        "发货慢",
        "配送慢",
        "快递慢",
        "延迟"
    ],

    "客服服务": [
        "客服态度差",
        "客服态度恶劣",
        "服务差",
        "不回复",
        "态度恶劣"
    ],

    "价格问题": [
        "价格贵",
        "太贵",
        "价格太高",
        "不值得",
        "性价比低"
    ],

    "售后问题": [
        "不退款",
        "退款困难",
        "退货困难",
        "售后差",
        "不处理"
    ]
}


def identify_risk_keywords(text):
    """
    根据风险关键词词典识别评论涉及的问题。

    Args:
        text: 评论文本

    Returns:
        list: 命中的风险问题类别
    """

    if pd.isna(text) or not str(text).strip():
        return []

    text = str(text)

    matched_categories = []

    for category, keywords in RISK_KEYWORDS.items():

        for keyword in keywords:

            if keyword in text:
                matched_categories.append(category)
                break

    return matched_categories


def analyze_risks(df):
    """
    分析评论中的舆情风险。

    Args:
        df: 已完成情感分析和主题聚类的数据

    Returns:
        dict: 舆情风险分析结果
    """

    if df.empty:
        raise ValueError("输入数据为空，无法进行风险分析")

    required_columns = [
        "content",
        "sentiment_label",
        "topic_id"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(f"数据中缺少必要字段：{column}")

    result_df = df.copy()

    # 识别每条评论涉及的风险问题
    result_df["risk_categories"] = result_df["content"].apply(
        identify_risk_keywords
    )

    # 筛选负面评论
    negative_df = result_df[
        result_df["sentiment_label"] == "负面"
    ].copy()

    total_count = len(result_df)
    negative_count = len(negative_df)

    # 计算负面评论占比
    negative_ratio = negative_count / total_count

    # 统计风险问题类别
    risk_counts = {}

    for categories in negative_df["risk_categories"]:

        for category in categories:

            risk_counts[category] = (
                risk_counts.get(category, 0) + 1
            )

    # 按数量从高到低排序
    risk_counts = dict(
        sorted(
            risk_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    # 统计负面评论的主题分布
    topic_distribution = (
        negative_df["topic_id"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # 风险等级判断（按负面评论占比，阈值说明见模块顶部）
    if negative_ratio >= HIGH_RISK_THRESHOLD:
        risk_level = "高风险"

    elif negative_ratio >= MEDIUM_RISK_THRESHOLD:
        risk_level = "中风险"

    else:
        risk_level = "低风险"

    return {
        "total_count": total_count,
        "negative_count": negative_count,
        "negative_ratio": round(negative_ratio, 4),
        "risk_level": risk_level,
        "risk_counts": risk_counts,
        "topic_distribution": topic_distribution,
        "negative_comments": negative_df,
        "result_df": result_df
    }
def save_risk_comments(
    risk_result,
    output_path="data/processed/risk_comments.csv"
):
    """
    保存负面风险评论明细。

    Args:
        risk_result: analyze_risks() 返回的分析结果
        output_path: CSV 文件保存路径

    Returns:
        str: 保存文件的路径
    """

    if not isinstance(risk_result, dict):
        raise ValueError("风险分析结果必须是字典")

    if "negative_comments" not in risk_result:
        raise ValueError("风险分析结果中缺少 negative_comments")

    negative_df = risk_result["negative_comments"]

    if not isinstance(negative_df, pd.DataFrame):
        raise ValueError("negative_comments 必须是 DataFrame")

    # 只保存需要的字段，并保留实际存在的字段
    columns = [
        "content",
        "sentiment_label",
        "topic_id",
        "risk_categories"
    ]

    available_columns = [
        column for column in columns
        if column in negative_df.columns
    ]

    if "content" not in available_columns:
        raise ValueError("风险评论数据中缺少 content 字段")

    save_df = negative_df[available_columns].copy()

    # 自动创建保存目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用 UTF-8-SIG，方便 Excel 正常打开中文 CSV
    save_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    return str(output_path)


def save_risk_summary(
    risk_result,
    output_path="reports/risk_summary.json"
):
    """
    保存舆情风险统计汇总。

    Args:
        risk_result: analyze_risks() 返回的分析结果
        output_path: JSON 文件保存路径

    Returns:
        str: 保存文件的路径
    """

    if not isinstance(risk_result, dict):
        raise ValueError("风险分析结果必须是字典")

    required_keys = [
        "total_count",
        "negative_count",
        "negative_ratio",
        "risk_level",
        "risk_counts",
        "topic_distribution"
    ]

    for key in required_keys:
        if key not in risk_result:
            raise ValueError(f"风险分析结果中缺少字段：{key}")

    # 只保存统计信息，不保存 DataFrame
    summary = {
        "total_count": risk_result["total_count"],
        "negative_count": risk_result["negative_count"],
        "negative_ratio": risk_result["negative_ratio"],
        "risk_level": risk_result["risk_level"],
        "risk_counts": risk_result["risk_counts"],
        "topic_distribution": risk_result["topic_distribution"]
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4
        )

    return str(output_path)