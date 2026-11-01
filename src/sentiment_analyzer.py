from snownlp import SnowNLP
import pandas as pd

from src.sentiment_model import load_sentiment_model, predict_sentiment
from src.sentiment_refiner import apply_rule_overrides, classify_comments


_MODEL = None
_MODEL_LOADED = False


def _get_model():
    """延迟加载一次已训练的情感模型，避免每次分析都重新读取文件。"""

    global _MODEL, _MODEL_LOADED

    if not _MODEL_LOADED:
        _MODEL = load_sentiment_model()
        _MODEL_LOADED = True

    return _MODEL


def analyze_sentiment(text):
    """
    分析单条评论的情感。

    判断顺序：
    明确情感规则 → 本地模型 → SnowNLP 回退。
    """

    if pd.isna(text) or not str(text).strip():
        return {
            "sentiment_score": None,
            "sentiment_label": "无法判断"
        }

    text = str(text).strip()

    # 1. 使用规则识别明确的正面、负面和中性表达
    rule_label = apply_rule_overrides(text)

    if rule_label is not None:
        score_map = {
            "正面": 0.95,
            "负面": 0.05,
            "中性": 0.50
        }

        return {
            "sentiment_score": score_map[rule_label],
            "sentiment_label": rule_label
        }

    # 2. 使用训练好的本地情感模型
    model = _get_model()
    score = None

    if model is not None:
        try:
            score = predict_sentiment(model, text)
        except Exception:
            score = None

    # 3. 模型不可用或预测失败时，回退到 SnowNLP
    if score is None:
        try:
            score = SnowNLP(text).sentiments
        except Exception:
            return {
                "sentiment_score": None,
                "sentiment_label": "无法判断"
            }

    # 4. 根据概率区间划分标签
    if score >= 0.65:
        label = "正面"
    elif score <= 0.40:
        label = "负面"
    else:
        label = "中性"

    return {
        "sentiment_score": round(float(score), 4),
        "sentiment_label": label
    }

def analyze_dataframe(df, text_column="content", refine_with_llm=False):
    """
    对 DataFrame 中的所有评论进行情感分析。

    refine_with_llm=True 时，会用大语言模型对模型不确定的评论精调。
    """

    if df.empty:
        raise ValueError("输入数据为空，无法进行情感分析")

    if text_column not in df.columns:
        raise ValueError(f"数据中缺少必要字段：{text_column}")

    result_df = df.copy()

    sentiment_results = result_df[text_column].apply(
        analyze_sentiment
    )

    result_df["sentiment_score"] = sentiment_results.apply(
        lambda x: x["sentiment_score"]
    )

    result_df["sentiment_label"] = sentiment_results.apply(
        lambda x: x["sentiment_label"]
    )

    if refine_with_llm:
        result_df = _refine_with_llm(result_df, text_column)

    return result_df


def _refine_with_llm(result_df, text_column):
    """对模型不确定的评论，用大语言模型按规则精调。"""

    scores = pd.to_numeric(
        result_df["sentiment_score"],
        errors="coerce"
    )

    uncertain_mask = scores.between(0.35, 0.65)

    if not uncertain_mask.any():
        return result_df

    indices = result_df.index[uncertain_mask].tolist()
    texts = result_df.loc[indices, text_column].tolist()

    try:
        labels = classify_comments(texts)
    except Exception:
        return result_df

    score_map = {
        "正面": 0.9,
        "负面": 0.1,
        "中性": 0.5
    }

    for index, label in zip(indices, labels):
        if label in score_map:
            result_df.loc[index, "sentiment_label"] = label
            result_df.loc[index, "sentiment_score"] = score_map[label]

    return result_df


def save_sentiment_results(
    df,
    output_path="data/processed/sentiment_results.csv"
):
    """
    保存情感分析结果。

    Args:
        df: 已完成情感分析的 DataFrame
        output_path: 结果保存路径

    Returns:
        str: 保存文件的路径
    """

    import os

    if df.empty:
        raise ValueError("情感分析结果为空，无法保存")

    required_columns = ["sentiment_score", "sentiment_label"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"缺少必要字段：{column}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"情感分析结果已保存至：{output_path}")

    return output_path
