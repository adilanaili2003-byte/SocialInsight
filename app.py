import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.agents.workflow import run_analysis
from src.data_processor import load_csv
from src.report_generator import generate_report, save_report


# 加载项目根目录下的 .env，用于展示页面顶部的运行状态。
load_dotenv(Path(__file__).resolve().parent / ".env")

API_KEY_SET = bool(os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "deepseek-v4-pro")


def topic_name(topic_id, keywords):
    """用主题编号加前两个关键词，生成可读的主题标签。"""

    if keywords:
        return f"主题 {topic_id} · {'/'.join(keywords[:2])}"

    return f"主题 {topic_id}"


# =========================
# 1. 页面基本配置
# =========================

st.set_page_config(
    page_title="SocialInsight 舆情分析系统",
    page_icon="📊",
    layout="wide"
)


# =========================
# 2. Session State 初始化
# =========================

st.session_state.setdefault("analysis_result", None)
st.session_state.setdefault("analysis_error", None)
st.session_state.setdefault("report_error", None)
st.session_state.setdefault("uploaded_file_id", None)


# =========================
# 3. 页面顶部：产品名称 / 项目简介 / 运行状态
# =========================

header_left, header_right = st.columns([3, 1], vertical_alignment="center")

with header_left:
    st.title("📊 SocialInsight 舆情分析系统")
    st.caption(
        "基于自然语言处理与大语言模型的社交媒体舆情分析平台，"
        "上传 CSV 评论数据即可完成清洗、情感、主题、风险与 AI 报告分析。"
    )

with header_right:
    if API_KEY_SET:
        st.success(
            f"就绪 · {MODEL_NAME}",
            icon=":material/check_circle:",
            title="运行状态"
        )
    else:
        st.warning(
            "请检查 .env 配置",
            icon=":material/warning:",
            title="运行状态"
        )

st.divider()


# =========================
# 4. 左侧边栏：数据上传
# =========================

with st.sidebar:
    st.header("数据上传")

    uploaded_file = st.file_uploader(
        "选择 CSV 评论数据文件",
        type=["csv"],
        help="CSV 文件必须包含 content 字段，表示评论内容。"
    )

    st.caption(
        "支持 UTF-8-SIG 与 GBK 编码；"
        "系统会自动删除空评论和重复评论。"
    )


# =========================
# 5. 新文件上传时清理上一次的结果
# =========================

if uploaded_file is not None:
    file_id = getattr(uploaded_file, "file_id", None)
    current_file_id = (file_id, uploaded_file.name, uploaded_file.size)

    if st.session_state["uploaded_file_id"] != current_file_id:
        st.session_state["uploaded_file_id"] = current_file_id
        st.session_state["analysis_result"] = None
        st.session_state["analysis_error"] = None
        st.session_state["report_error"] = None


# =========================
# 6. 读取并校验上传文件（与后端流水线使用相同的读取逻辑）
# =========================

preview_df = None
load_error = None
has_content = False

if uploaded_file is not None:
    try:
        preview_df = load_csv(uploaded_file)
        has_content = "content" in preview_df.columns
    except Exception as e:
        load_error = str(e)


# =========================
# 7. 左侧边栏：分析操作
# =========================

with st.sidebar:
    st.divider()
    st.header("分析操作")
    st.caption(
        "点击下方按钮后，系统会依次完成数据清洗、情感分析、"
        "主题聚类、风险识别，并调用大语言模型生成舆情报告。"
    )

    analyze_disabled = uploaded_file is None or not has_content

    if st.button(
        "开始舆情分析",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
        disabled=analyze_disabled
    ):

        with st.spinner("正在进行舆情分析，请稍候..."):
            try:
                # 重置文件读取位置，确保工作流能重新读取上传内容。
                uploaded_file.seek(0)

                result = run_analysis(
                    file=uploaded_file,
                    generate_ai_report=False,
                    save_results=True,
                    refine_sentiment=False
                )

                # AI 报告单独生成：即使报告生成失败，也保留前面的分析结果。
                try:
                    report = generate_report(result["analysis_result"])
                    save_report(report)
                except Exception as e:
                    st.session_state["report_error"] = str(e)
                    report = None
                else:
                    st.session_state["report_error"] = None

                result["report"] = report
                st.session_state["analysis_result"] = result
                st.session_state["analysis_error"] = None

                if report:
                    st.toast("舆情分析完成！", icon=":material/check_circle:")
                else:
                    st.toast(
                        "分析完成，但 AI 报告未生成",
                        icon=":material/warning:"
                    )

            except Exception as e:
                st.session_state["analysis_result"] = None
                st.session_state["analysis_error"] = str(e)
                st.session_state["report_error"] = None


# =========================
# 8. 主页面：错误提示
# =========================

analysis_error = st.session_state.get("analysis_error")

if analysis_error:
    st.error(f"分析失败：{analysis_error}")


# =========================
# 9. 主页面：原始数据预览
# =========================

if load_error is not None:
    st.error(f"文件读取失败：{load_error}")

elif preview_df is not None:
    st.success("文件上传成功！")
    st.subheader("原始数据预览")
    st.write(f"数据总量：{len(preview_df)} 条")

    st.dataframe(
        preview_df.head(10),
        hide_index=True,
        alt="上传 CSV 的前 10 条数据预览"
    )

    if not has_content:
        st.error(
            "CSV 文件缺少必要字段 content，请检查文件格式。"
        )

elif st.session_state.get("analysis_result") is None:
    st.info("请先在左侧上传 CSV 评论数据文件。")


# =========================
# 10. 主页面：分析结果
# =========================

if st.session_state.get("analysis_result") is not None:

    result = st.session_state["analysis_result"]
    risk_result = result["risk_result"]

    st.divider()
    st.header("分析结果")

    # 10.1 核心指标
    kpi_columns = st.columns(4, border=True)

    with kpi_columns[0]:
        st.metric("有效评论数量", risk_result["total_count"])

    with kpi_columns[1]:
        st.metric("负面评论数量", risk_result["negative_count"])

    with kpi_columns[2]:
        st.metric("负面评论比例", f"{risk_result['negative_ratio']:.2%}")

    with kpi_columns[3]:
        st.metric("当前风险等级", risk_result["risk_level"])

    # 10.2 情感分析可视化
    st.subheader("评论情感分布")

    sentiment_data = result["sentiment_data"]

    sentiment_order = ["正面", "中性", "负面"]

    sentiment_counts = (
        sentiment_data["sentiment_label"]
        .value_counts()
        .reindex(sentiment_order, fill_value=0)
    )

    sentiment_chart_data = pd.DataFrame({
        "情感类别": sentiment_order,
        "评论数量": [
            int(sentiment_counts[label])
            for label in sentiment_order
        ]
    })

    sentiment_chart = (
        alt.Chart(sentiment_chart_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "情感类别:N",
                sort=sentiment_order,
                axis=alt.Axis(
                    labelAngle=0,
                    labelFontSize=14,
                    title=None
                )
            ),
            y=alt.Y(
                "评论数量:Q",
                axis=alt.Axis(
                    title="评论数量",
                    tickMinStep=1
                )
            ),
            tooltip=["情感类别", "评论数量"]
        )
        .properties(height=350)
    )

    st.altair_chart(
        sentiment_chart,
        alt="正面、中性、负面评论数量分布柱状图"
    )

    sentiment_columns = st.columns(3, border=True)

    with sentiment_columns[0]:
        st.metric("正面评论", int(sentiment_counts["正面"]))

    with sentiment_columns[1]:
        st.metric("中性评论", int(sentiment_counts["中性"]))

    with sentiment_columns[2]:
        st.metric("负面评论", int(sentiment_counts["负面"]))

    # 10.3 评论分析明细
    st.subheader("评论分析明细")

    sentiment_display_columns = [
        column
        for column in [
            "content",
            "sentiment_score",
            "sentiment_label",
            "topic_id"
        ]
        if column in sentiment_data.columns
    ]

    st.dataframe(
        sentiment_data[sentiment_display_columns],
        hide_index=True,
        alt="每条评论的情感得分、情感标签和主题编号明细"
    )

    st.caption(
        "说明：情感由规则引擎、TF-IDF + 逻辑回归模型、SnowNLP "
        "三级流水线综合判断；主题编号由 TF-IDF 与 KMeans 聚类得到。"
    )

    # 10.4 主题聚类可视化
    st.subheader("评论主题分布")

    topic_data = result["topic_data"]

    topic_counts = (
        topic_data["topic_id"]
        .value_counts()
        .sort_index()
    )

    topic_chart_data = pd.DataFrame({
        "主题": [
            topic_name(
                topic_id,
                result["topic_keywords"].get(topic_id, [])
            )
            for topic_id in topic_counts.index
        ],
        "评论数量": topic_counts.values
    })

    topic_chart = (
        alt.Chart(topic_chart_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "主题:N",
                sort=None,
                axis=alt.Axis(
                    labelAngle=0,
                    labelFontSize=14,
                    title=None
                )
            ),
            y=alt.Y(
                "评论数量:Q",
                axis=alt.Axis(
                    title="评论数量",
                    tickMinStep=1
                )
            ),
            tooltip=["主题", "评论数量"]
        )
        .properties(height=350)
    )

    st.altair_chart(
        topic_chart,
        alt="各主题评论数量分布柱状图"
    )

    # 10.5 主题关键词展示
    st.subheader("各主题核心关键词")

    topic_keywords = result["topic_keywords"]
    topic_ids = sorted(topic_keywords.keys())

    topic_columns = st.columns(len(topic_ids), border=True)

    for index, topic_id in enumerate(topic_ids):

        with topic_columns[index]:
            keywords = topic_keywords[topic_id]

            st.markdown(f"**{topic_name(topic_id, keywords)}**")

            comment_count = int(
                (topic_data["topic_id"] == topic_id).sum()
            )

            st.metric("评论数量", comment_count)

            if keywords:
                st.markdown("、".join(keywords))
            else:
                st.caption("暂无关键词")

    # 10.6 主题评论明细
    st.subheader("各主题评论明细")

    for topic_id in topic_ids:

        with st.expander(
            f"查看 {topic_name(topic_id, topic_keywords[topic_id])} 的评论"
        ):

            topic_comments = topic_data[
                topic_data["topic_id"] == topic_id
            ]

            topic_display_columns = [
                column
                for column in [
                    "content",
                    "sentiment_label",
                    "sentiment_score"
                ]
                if column in topic_comments.columns
            ]

            st.dataframe(
                topic_comments[topic_display_columns],
                hide_index=True,
                alt=f"主题 {topic_id} 的评论明细"
            )

    # 10.7 舆情风险分析
    st.header("舆情风险分析")

    risk_counts = risk_result["risk_counts"]

    st.subheader("风险类别分布")

    if risk_counts:

        risk_chart_data = pd.DataFrame({
            "风险类别": list(risk_counts.keys()),
            "评论数量": list(risk_counts.values())
        })

        risk_chart = (
            alt.Chart(risk_chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "风险类别:N",
                    sort=None,
                    axis=alt.Axis(
                        labelAngle=0,
                        labelFontSize=13,
                        title=None
                    )
                ),
                y=alt.Y(
                    "评论数量:Q",
                    axis=alt.Axis(
                        title="评论数量",
                        tickMinStep=1
                    )
                ),
                tooltip=["风险类别", "评论数量"]
            )
            .properties(height=350)
        )

        st.altair_chart(
            risk_chart,
            alt="各风险类别评论数量分布柱状图"
        )

    else:
        st.info("当前没有识别到明确的风险类别。")

    # 10.8 风险评论明细
    st.subheader("风险评论明细")

    negative_comments = risk_result["negative_comments"]

    if not negative_comments.empty:

        risk_display_columns = [
            column
            for column in [
                "content",
                "sentiment_score",
                "sentiment_label",
                "risk_categories",
                "topic_id"
            ]
            if column in negative_comments.columns
        ]

        st.dataframe(
            negative_comments[risk_display_columns],
            hide_index=True,
            alt="负面评论及其风险类别明细"
        )

    else:
        st.success("当前没有识别到负面评论。")

    # 10.9 风险等级说明
    st.subheader("当前舆情风险评估")

    risk_level = risk_result["risk_level"]
    negative_ratio = risk_result["negative_ratio"]
    negative_count = risk_result["negative_count"]
    total_count = risk_result["total_count"]

    risk_messages = {
        "高风险": (
            "🔴 当前舆情风险较高，建议重点关注负面评论，"
            "及时排查问题并制定应对措施。"
        ),
        "中风险": (
            "🟠 当前存在一定舆情风险，建议关注负面评论集中反映的问题，"
            "并持续观察舆情变化。"
        ),
        "低风险": (
            "🟢 当前舆情风险较低，建议继续关注用户反馈，"
            "及时处理新出现的问题。"
        )
    }

    st.info(
        f"当前风险等级：{risk_level}\n\n"
        f"负面评论数量：{negative_count} 条\n\n"
        f"负面评论比例：{negative_ratio:.2%}\n\n"
        f"统计评论总数：{total_count} 条\n\n"
        f"{risk_messages.get(risk_level, '暂无风险等级说明。')}"
    )

    st.caption(
        "风险等级按负面评论占比判断：占比 ≥ 50% 判高风险，≥ 20% 判中风险，"
        "其余判低风险。该规则属于项目初期简化标准，仅以负面占比为信号，"
        "未综合风险类别严重程度与情绪强度，并非经过真实业务验证的行业标准。"
    )

    # 10.10 AI 舆情报告展示
    st.header("AI 舆情分析报告")

    report = result.get("report")

    if report:

        st.markdown(
            "以下报告由 DeepSeek-V4-Pro 根据本次分析结果生成。"
        )

        with st.container(border=True):
            st.markdown(report)

        st.download_button(
            label="下载 AI 舆情分析报告",
            data=report,
            file_name="SocialInsight_舆情分析报告.md",
            mime="text/markdown",
            type="primary",
            icon=":material/download:",
            width="stretch"
        )

    else:
        report_error = st.session_state.get("report_error")

        if report_error:
            st.warning(f"AI 舆情报告生成失败：{report_error}")
        else:
            st.warning(
                "本次分析未生成 AI 舆情报告，"
                "请检查大语言模型 API 配置或重新运行分析。"
            )

    # 10.11 完整分析结果下载
    st.header("完整分析结果导出")

    full_results = result["risk_result"]["result_df"]

    csv_data = full_results.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="下载完整评论分析结果",
        data=csv_data,
        file_name="SocialInsight_完整分析结果.csv",
        mime="text/csv",
        type="primary",
        icon=":material/download:",
        width="stretch"
    )

    st.caption(
        "导出文件包含评论原文、情感得分、情感标签、主题编号及风险类别等分析信息。"
    )
