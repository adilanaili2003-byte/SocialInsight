import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


SYSTEM_PROMPT = (
    "你是专业的中文评论情感标注员。请严格遵循以下标准：\n"
    "1. 【正面】表达喜爱、认可、推荐、觉得有效、会回购等。含“欲扬先抑”"
    "（开头吐槽缺点但最终落脚点是夸奖或推荐）判为正面。\n"
    "2. 【负面】表达失望、避雷、过敏、无效、觉得被欺骗、太贵且不值等。"
    "含“反讽/阴阳怪气”（表面用词好但实际极度不满）判为负面。\n"
    "3. 【中性】客观陈述事实、单纯提问（如“这个在哪买？”）、"
    "或正负面评价各占一半且无明显倾向的纯对比。\n"
    "必须结合整段话的最终语境，不要被单个词汇骗了。例如“除了贵没毛病”是正面，"
    "“除了便宜一无是处”是负面。警惕网络黑话：“真香”“绝绝子”是正面，"
    "“踩大雷”“拔草”通常是负面。"
)


# ==============================
# 通用中文情感规则
# ==============================

POSITIVE_SIGNALS = [
    "真香", "绝绝子", "好喝", "好吃", "很好吃",
    "很香", "满意", "喜欢", "推荐", "回购",
    "值得购买", "值得推荐", "很不错", "非常不错",
    "赞不绝口", "质量很好", "服务很好",
    "效果很好", "体验很好", "非常满意",
    "非常好用", "特别好用", "值得入手",
    "物超所值", "性价比高", "很方便",
    "很有效", "很喜欢", "太棒了"
]

NEGATIVE_SIGNALS = [
    "踩大雷", "拔草", "避雷", "难吃", "难喝",
    "失望", "后悔", "不满意", "不好吃", "不好喝",
    "不太好吃", "不是很好吃", "太差", "很差",
    "太贵", "不值得", "不推荐", "不会再买",
    "不会回购", "再也不买", "无效", "过敏",
    "被骗", "欺骗", "想吐", "不能用",
    "完全不能用", "质量差", "态度差",
    "太慢了", "非常难用", "不好用",
    "不值得购买", "不值得推荐", "太失望了",
    "体验很差", "非常糟糕", "令人失望"
]

# 明确的正面固定句式
POSITIVE_PATTERNS = [
    ("除了", "没毛病"),
]

# 明确的负面固定句式
NEGATIVE_PATTERNS = [
    ("除了", "一无是处"),
]

# 转折词
CONTRAST_WORDS = [
    "但是", "不过", "可是", "然而", "就是"
]

# 否定表达
NEGATION_PREFIXES = [
    "不", "没", "并不", "并没有",
    "不是", "并非", "谈不上"
]

# 明确中性表达
NEUTRAL_SIGNALS = [
    "不知道好不好",
    "还不知道",
    "不清楚",
    "还没试",
    "还没用",
    "还没吃",
    "还没喝",
    "没有评价",
    "一般般",
    "没什么特别",
    "没有特别好也没有特别差",
    "名字跟我们家名字一样",
    "外卖员名字跟我们家名字一样"
]

# 客观事实描述
FACTUAL_SIGNALS = [
    "收到商品",
    "收到了商品",
    "今天收到",
    "刚刚收到",
    "包装完好",
    "购买了一件",
    "买了一杯",
    "买了一份",
    "价格十五元",
    "价格十元",
    "价格二十元",
    "价格是",
    "价格为",
    "一共是",
    "名字一样",
    "数量为",
    "订单编号",
    "今天下单",
    "已经发货",
    "正在配送"
]

# 可能表示反讽的事件表达
NEGATIVE_CONTEXT_SIGNALS = [
    "等了两个小时",
    "等了半天",
    "迟到了",
    "又坏了",
    "不能用",
    "出问题了",
    "出了问题",
    "一直没到",
    "没有送到",
    "又涨价了",
    "完全失效",
]


def _contains_signal(text, signals):
    """检查文本是否包含指定表达。"""
    return any(signal in text for signal in signals)


def _get_sentiment_from_text(text):
    """
    识别文本中明确的正面或负面表达。

    对正面词语前面的否定表达进行简单处理。
    正负面信号同时存在时，不强行判断。
    """

    positive_found = []
    negative_found = []

    for signal in POSITIVE_SIGNALS:
        position = text.find(signal)

        if position == -1:
            continue

        prefix = text[max(0, position - 3):position]

        # 处理“不好吃”“不是很好”等否定表达
        if any(prefix.endswith(word) for word in NEGATION_PREFIXES):
            negative_found.append(signal)
        else:
            positive_found.append(signal)

    for signal in NEGATIVE_SIGNALS:
        if signal in text:
            negative_found.append(signal)

    has_positive = bool(positive_found)
    has_negative = bool(negative_found)

    if has_positive and not has_negative:
        return "正面"

    if has_negative and not has_positive:
        return "负面"

    return None


def _get_contrast_result(text):
    """
    识别转折句。

    转折后的明确情感倾向优先；
    转折后无法判断时，不强行给出结果。
    """

    positions = [
        (text.rfind(word), word)
        for word in CONTRAST_WORDS
        if word in text
    ]

    if not positions:
        return None

    position, word = max(positions, key=lambda item: item[0])

    before_text = text[:position]
    after_text = text[position + len(word):]

    after_label = _get_sentiment_from_text(after_text)

    if after_label is not None:
        return after_label

    before_label = _get_sentiment_from_text(before_text)

    if before_label is not None:
        return before_label

    return None


def apply_rule_overrides(text):
    """
    通用中文评论情感规则。

    判断顺序：
    固定句式 → 转折表达 → 反讽特征 →
    明确情感表达 → 中性表达 → 客观事实。

    无法确定时返回 None，交给本地模型。
    """

    text = str(text).strip()

    if not text:
        return None
        # 对尚未形成明确评价的表达，优先判定为中性
    uncertain_opinion_patterns = [
        "还不知道好不好",
        "不知道好不好",
        "还不清楚好不好",
        "不确定好不好"
    ]

    if _contains_signal(text, uncertain_opinion_patterns):
        return "中性"

    # 1. 明确固定句式
    for prefix, suffix in POSITIVE_PATTERNS:
        if prefix in text and suffix in text:
            return "正面"

    for prefix, suffix in NEGATIVE_PATTERNS:
        if prefix in text and suffix in text:
            return "负面"

    # 2. 识别转折句
    contrast_result = _get_contrast_result(text)

    if contrast_result is not None:
        return contrast_result

    # 3. 识别有明显负面事件支撑的反讽
    if (
        _contains_signal(text, ["真是太好了", "真不错", "真棒"])
        and _contains_signal(text, NEGATIVE_CONTEXT_SIGNALS)
    ):
        return "负面"

    # 4. 网络黑话
    if _contains_signal(text, ["真香", "绝绝子"]):
        return "正面"

    if _contains_signal(text, ["踩大雷", "拔草", "避雷"]):
        return "负面"

    # 5. 明确正负面表达
    sentiment = _get_sentiment_from_text(text)

    if sentiment is not None:
        return sentiment

    # 6. 单纯提问
    if text.endswith(("?", "？")):
        return "中性"

    if text.startswith(("请问", "想问一下", "问一下")):
        return "中性"

    if any(
        phrase in text
        for phrase in ["在哪里买", "在哪买", "哪里买", "怎么买"]
    ):
        return "中性"

    # 7. 明确中性表达
    if _contains_signal(text, NEUTRAL_SIGNALS):
        return "中性"

    # 8. 客观事实描述
    if _contains_signal(text, FACTUAL_SIGNALS):
        return "中性"

    # 9. 无法确定，交给本地模型
    return None

def classify_comments(texts, batch_size=40):
    """用大语言模型按规则批量判断评论情感，返回与输入等长的标签列表。"""

    if not texts:
        return []

    labels = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        labels.extend(_classify_chunk(chunk))

    return labels


def _classify_chunk(chunk):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL")

    if not api_key or not base_url or not model:
        raise ValueError("未配置大语言模型 API，无法进行情感精调")

    lines = "\n".join(
        f"{index}. {text}"
        for index, text in enumerate(chunk, start=1)
    )

    user_prompt = (
        "请判断以下评论的情感类别（正面/负面/中性）。\n"
        "只输出一个 JSON 数组，每个元素形如 {\"id\": 序号, \"label\": \"正面\"}。\n"
        "不要输出任何解释或其他文字。\n\n"
        f"{lines}"
    )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=45,
        max_retries=1
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content or ""
    items = _parse_json_array(content)

    label_by_index = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        try:
            index = int(item.get("id", -1))
            label = str(item.get("label", "")).strip()
        except (AttributeError, TypeError, ValueError):
            continue

        if index >= 1 and label in ("正面", "负面", "中性"):
            label_by_index[index - 1] = label

    return [label_by_index.get(i) for i in range(len(chunk))]


def _parse_json_array(text):
    text = text.strip()

    if text.startswith("```"):
        parts = text.split("```")

        if len(parts) >= 3:
            text = parts[1].strip()

            if text.startswith("json"):
                text = text[4:].strip()

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("模型未返回有效的 JSON 数组")

    return json.loads(text[start:end + 1])
