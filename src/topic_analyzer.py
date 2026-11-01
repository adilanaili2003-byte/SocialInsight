import jieba
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# 常见中文停用词
STOP_WORDS = {
    "的", "了", "是", "我", "也", "都", "就",
    "很", "非常", "一个", "这个", "那个",
    "但是", "而且", "因为", "所以", "如果",
    "没有", "不是", "可以", "感觉", "还是",
    "他们", "我们", "你们", "它们", "什么",
    "怎么", "这样", "那样", "自己", "比较",
    "一些", "现在", "真的", "东西", "不错"
}


def tokenize_text(text):
    """
    对单条评论进行中文分词。

    Args:
        text: 评论文本

    Returns:
        list: 分词后的有效词语
    """

    if pd.isna(text) or not str(text).strip():
        return []

    words = jieba.lcut(str(text))

    # 去除停用词和无意义字符
    words = [
        word.strip()
        for word in words
        if word.strip()
        and word.strip() not in STOP_WORDS
        and len(word.strip()) > 1
    ]

    return words


def extract_keywords(df, text_column="content", top_n=20):
    """
    提取评论中的高频关键词。

    Args:
        df: 评论数据
        text_column: 评论文本字段
        top_n: 返回关键词数量

    Returns:
        list: 高频关键词及出现次数
    """

    if df.empty:
        raise ValueError("输入数据为空，无法提取关键词")

    if text_column not in df.columns:
        raise ValueError(f"数据中缺少必要字段：{text_column}")

    all_words = []

    for text in df[text_column]:
        all_words.extend(tokenize_text(text))

    word_counts = Counter(all_words)

    return word_counts.most_common(top_n)


def cluster_topics(
    df,
    text_column="content",
    n_clusters=3
):
    """
    使用 TF-IDF 和 KMeans 进行评论主题聚类。

    Args:
        df: 评论数据
        text_column: 评论文本字段
        n_clusters: 聚类数量

    Returns:
        DataFrame: 添加主题编号后的数据
        vectorizer: TF-IDF 向量化器
        model: KMeans 模型
    """

    if df.empty:
        raise ValueError("输入数据为空，无法进行主题聚类")

    if text_column not in df.columns:
        raise ValueError(f"数据中缺少必要字段：{text_column}")

    if not 1 <= n_clusters <= len(df):
        raise ValueError("聚类数量必须大于等于1且不能超过评论数量")

    result_df = df.copy()

    # 评论分词
    tokenized_texts = [
        " ".join(tokenize_text(text))
        for text in result_df[text_column]
    ]

    # TF-IDF 特征提取
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        max_features=1000
    )

    tfidf_matrix = vectorizer.fit_transform(tokenized_texts)

    if tfidf_matrix.shape[1] == 0:
        raise ValueError("未能提取有效文本特征，请检查评论内容")

    # KMeans 聚类
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    cluster_labels = model.fit_predict(tfidf_matrix)

    result_df["topic_id"] = cluster_labels

    return result_df, vectorizer, model


def get_topic_keywords(
    vectorizer,
    model,
    top_n=5
):
    """
    获取每个主题的代表性关键词。

    Args:
        vectorizer: TF-IDF 向量化器
        model: KMeans 模型
        top_n: 每个主题返回的关键词数量

    Returns:
        dict: 各主题对应的关键词
    """

    feature_names = vectorizer.get_feature_names_out()

    topic_keywords = {}

    for topic_id, center in enumerate(model.cluster_centers_):

        # 找出当前主题权重最高的词语
        top_indices = center.argsort()[::-1][:top_n]

        keywords = [
            feature_names[index]
            for index in top_indices
        ]

        topic_keywords[topic_id] = keywords

    return topic_keywords
def save_topic_results(
    df,
    output_path="data/processed/topic_results.csv"
):
    """
    保存主题聚类结果。

    Args:
        df: 已完成主题聚类的 DataFrame
        output_path: 保存路径

    Returns:
        str: 保存文件路径
    """
    import os

    if df.empty:
        raise ValueError("主题分析结果为空，无法保存")

    if "topic_id" not in df.columns:
        raise ValueError("数据中缺少必要字段：topic_id")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"主题分析结果已保存至：{output_path}")

    return output_path