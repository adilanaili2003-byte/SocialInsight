import json
from pathlib import Path

from src.report_generator import (
    build_report_prompt,
    save_report
)


def create_test_analysis():
    """构造模拟舆情分析数据。"""

    return {
        "total_count": 100,
        "negative_count": 30,
        "negative_ratio": 0.3,
        "risk_level": "中风险",
        "risk_counts": {
            "产品质量": 12,
            "物流服务": 8,
            "客服服务": 5
        },
        "topic_distribution": {
            0: 12,
            1: 10,
            2: 8
        },
        "topic_keywords": {
            0: ["产品", "质量", "使用"],
            1: ["物流", "配送", "发货"],
            2: ["客服", "售后", "服务"]
        }
    }


def test_build_report_prompt():
    """测试报告提示词构建。"""

    data = create_test_analysis()

    prompt = build_report_prompt(data)

    assert "100" in prompt
    assert "30.00%" in prompt
    assert "中风险" in prompt
    assert "产品质量" in prompt
    assert "改进建议" in prompt

    print("报告提示词构建测试通过！")


def test_build_report_prompt_invalid():
    """测试缺少必要字段时的异常处理。"""

    try:
        build_report_prompt({})
    except ValueError:
        print("提示词异常处理测试通过！")
    else:
        raise AssertionError("缺少必要字段时没有触发 ValueError")


def test_save_report(tmp_path):
    """测试 Markdown 报告保存。"""

    report = """# 舆情分析报告

## 1. 舆情整体概况

本次共分析100条评论。

## 2. 风险分析

当前风险等级为中风险。

## 3. 改进建议

建议关注产品质量问题。
"""

    output_path = tmp_path / "nested" / "test_report.md"

    saved_path = save_report(
        report,
        output_path
    )

    assert Path(saved_path).exists()

    with open(saved_path, "r", encoding="utf-8") as file:
        saved_content = file.read()

    assert saved_content == report
    assert "舆情分析报告" in saved_content
    assert "产品质量" in saved_content

    print("Markdown 报告保存测试通过！")


def test_save_report_invalid():
    """测试空报告异常处理。"""

    try:
        save_report("")
    except ValueError:
        print("报告保存异常处理测试通过！")
    else:
        raise AssertionError("空报告没有触发 ValueError")


if __name__ == "__main__":
    test_build_report_prompt()
    test_build_report_prompt_invalid()
    test_save_report(Path("test_temp_report"))
    test_save_report_invalid()

    print("\n报告生成模块基础测试全部完成！")