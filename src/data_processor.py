import pandas as pd
import re
from typing import Dict, Tuple


def load_csv(file) -> pd.DataFrame:
    """
    读取 CSV 评论数据。

    参数：
        file: CSV 文件路径或文件对象

    返回：
        df: 原始评论数据
    """

    try:
        df = pd.read_csv(file, encoding="utf-8-sig")

    except UnicodeDecodeError:
        # 如果 UTF-8 编码读取失败，尝试 GBK
        try:
            if hasattr(file, "seek"):
                file.seek(0)

            df = pd.read_csv(file, encoding="gbk")

        except Exception as e:
            raise ValueError(f"CSV 文件编码错误：{e}")

    except pd.errors.EmptyDataError:
        raise ValueError("CSV 文件为空。")

    except Exception as e:
        raise ValueError(f"CSV 文件读取失败：{e}")

    if df.empty:
        raise ValueError("CSV 文件中没有数据。")

    return df


def clean_text(text):
    """
    清洗单条评论文本。

    处理内容：
    1. 删除空值
    2. 统一数据类型
    3. 删除多余空格和换行符
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # 删除换行符和制表符
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # 将连续空格替换成一个空格
    text = re.sub(r"\s+", " ", text)

    # 删除首尾空格
    text = text.strip()

    return text


def clean_dataframe(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """
    清洗评论 DataFrame。

    参数：
        df: 原始评论数据

    返回：
        cleaned_df: 清洗后的数据
        stats: 数据清洗统计信息
    """

    # 检查数据是否为空
    if df is None or df.empty:
        raise ValueError("上传的数据为空，请检查 CSV 文件。")

    # 记录原始数据量
    original_count = len(df)

    # 检查必要字段
    if "content" not in df.columns:
        raise ValueError("CSV 文件缺少必要字段：content")

    # 复制数据，避免修改原始数据
    cleaned_df = df.copy()

    # 清洗评论文本
    cleaned_df["content"] = cleaned_df["content"].apply(clean_text)

    # 删除空评论
    cleaned_df = cleaned_df[
        cleaned_df["content"] != ""
    ].copy()

    after_empty_count = len(cleaned_df)

    # 删除重复评论
    cleaned_df = cleaned_df.drop_duplicates(
        subset=["content"],
        keep="first"
    ).copy()

    final_count = len(cleaned_df)

    # 重置索引
    cleaned_df = cleaned_df.reset_index(drop=True)

    # 统计清洗结果
    stats = {
        "original_count": original_count,
        "after_empty_removal": after_empty_count,
        "final_count": final_count,
        "removed_empty": original_count - after_empty_count,
        "removed_duplicates": after_empty_count - final_count,
        "removed_total": original_count - final_count
    }

    return cleaned_df, stats
def save_processed_data(df, output_path="data/processed/cleaned_comments.csv"):
    """
    保存清洗后的评论数据。

    Args:
        df: 清洗后的 DataFrame
        output_path: 保存路径

    Returns:
        str: 文件保存路径
    """
    import os

    if df.empty:
        raise ValueError("清洗后的数据为空，无法保存")

    # 自动创建目标文件夹
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存 CSV，使用 UTF-8-SIG 兼容 Excel
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"清洗后的数据已保存至：{output_path}")

    return output_path