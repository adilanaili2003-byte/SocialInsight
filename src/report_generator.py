import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# 加载项目根目录下的 .env 文件
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def build_report_prompt(analysis_result):
    """
    根据舆情分析结果构建报告提示词。

    Args:
        analysis_result: 完整舆情分析数据

    Returns:
        str: 报告生成提示词
    """

    required_keys = [
        "total_count",
        "negative_count",
        "negative_ratio",
        "risk_level",
        "risk_counts",
        "topic_distribution",
        "topic_keywords"
    ]

    for key in required_keys:
        if key not in analysis_result:
            raise ValueError(f"分析数据缺少必要字段：{key}")

    prompt = f"""
你是一名专业的社交媒体舆情分析师。

请根据下面提供的真实分析数据，生成一份结构清晰、内容客观的舆情分析报告。

一、数据概况
评论总数：{analysis_result['total_count']}
负面评论数量：{analysis_result['negative_count']}
负面评论比例：{analysis_result['negative_ratio']:.2%}
当前风险等级：{analysis_result['risk_level']}

二、风险类别统计
{analysis_result['risk_counts']}

三、负面评论主题分布
{analysis_result['topic_distribution']}

四、主题关键词
{analysis_result['topic_keywords']}

请按照以下结构生成报告：

# 舆情分析报告

## 1. 舆情整体概况
介绍本次评论数据的整体情况。

## 2. 情感分析
分析正面、负面评论情况。
注意：只能根据提供的数据进行分析，不得编造正面评论数量。

## 3. 主要讨论主题
结合主题关键词和主题分布，概括用户主要讨论的问题。

## 4. 舆情风险分析
分析风险等级、主要风险类别及其可能产生的影响。

## 5. 改进建议
针对识别出的风险问题提出具体、可执行的建议。

## 6. 总结
概括本次舆情分析的主要发现。

特别要求：
1. 所有统计数字必须来自提供的数据。
2. 不得编造评论内容、用户反馈或统计结果。
3. 不得将相关性描述为因果关系。
4. 如果样本量较小，必须说明分析结果存在局限性。
5. 如果某项统计数据不足以支持判断，应明确说明。
6. 使用客观、规范的中文表达。
7. 直接输出 Markdown 格式报告，不要输出额外说明。
"""

    return prompt


def generate_report(analysis_result):
    """
    调用 DeepSeek API 生成舆情分析报告。

    Args:
        analysis_result: 完整舆情分析数据

    Returns:
        str: Markdown 格式报告
    """

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise ValueError("未配置 OPENAI_API_KEY")

    if not base_url:
        raise ValueError("未配置 OPENAI_BASE_URL")

    if not model:
        raise ValueError("未配置 OPENAI_MODEL")

    prompt = build_report_prompt(analysis_result)

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一名严谨的中文舆情分析专家。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        report = response.choices[0].message.content

        if not report or not report.strip():
            raise ValueError("模型返回的报告内容为空")

        return report.strip()

    except Exception as e:
        raise RuntimeError(f"舆情报告生成失败：{e}") from e


def save_report(
    report,
    output_path="reports/sentiment_report.md"
):
    """
    保存 Markdown 格式的舆情报告。

    Args:
        report: 模型生成的报告
        output_path: 保存路径

    Returns:
        str: 文件保存路径
    """

    if not isinstance(report, str) or not report.strip():
        raise ValueError("报告内容不能为空")

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)

    return str(output_path)