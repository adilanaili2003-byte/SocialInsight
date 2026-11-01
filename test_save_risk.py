import json
from pathlib import Path

import pandas as pd

from src.risk_analyzer import (
    analyze_risks,
    save_risk_comments,
    save_risk_summary
)


def create_test_data():
    """构造测试评论数据。"""

    data = {
        "content": [
            "产品质量太差了，已经损坏",
            "物流太慢了，等了很久",
            "客服态度恶劣，不处理问题",
            "产品质量很好，使用体验不错"
        ],
        "sentiment_label": [
            "负面",
            "负面",
            "负面",
            "正面"
        ],
        "topic_id": [0, 1, 2, 0]
    }

    return pd.DataFrame(data)


def test_save_risk_comments(tmp_path):
    """测试风险评论 CSV 保存。"""

    df = create_test_data()
    result = analyze_risks(df)

    output_path = tmp_path / "nested" / "risk_comments.csv"

    saved_path = save_risk_comments(
        result,
        output_path
    )

    assert Path(saved_path).exists()

    saved_df = pd.read_csv(
        saved_path,
        encoding="utf-8-sig"
    )

    assert len(saved_df) == 3

    assert "content" in saved_df.columns
    assert "sentiment_label" in saved_df.columns
    assert "topic_id" in saved_df.columns
    assert "risk_categories" in saved_df.columns

    assert saved_df.iloc[0]["content"] == "产品质量太差了，已经损坏"

    print("\n风险评论 CSV 保存测试通过！")


def test_save_risk_summary(tmp_path):
    """测试风险统计 JSON 保存。"""

    df = create_test_data()
    result = analyze_risks(df)

    output_path = tmp_path / "nested" / "risk_summary.json"

    saved_path = save_risk_summary(
        result,
        output_path
    )

    assert Path(saved_path).exists()

    with open(saved_path, "r", encoding="utf-8") as file:
        summary = json.load(file)

    assert summary["total_count"] == 4
    assert summary["negative_count"] == 3
    assert summary["negative_ratio"] == 0.75
    assert summary["risk_level"] == "高风险"

    assert "risk_counts" in summary
    assert "topic_distribution" in summary

    print("\n风险统计 JSON 保存测试通过！")


def test_save_risk_summary_invalid_input():
    """测试无效输入处理。"""

    try:
        save_risk_summary({})
    except ValueError:
        print("\n无效输入测试通过！")
    else:
        raise AssertionError("无效输入没有触发 ValueError")


if __name__ == "__main__":
    test_dir = Path("test_temp_risk")

    test_save_risk_comments(test_dir)
    test_save_risk_summary(test_dir)
    test_save_risk_summary_invalid_input()

    print("\n全部风险结果保存测试完成！")