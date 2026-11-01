from src.sentiment_analyzer import analyze_sentiment


test_cases = [
    # 正面评论
    ("味道很香，非常好喝", "正面"),
    ("好喝，下次还会点", "正面"),
    ("服务态度很好，非常满意", "正面"),
    ("真的很不错，值得推荐", "正面"),
    ("点了很多次，每次都很满意", "正面"),
    ("虽然有点贵，但是味道真的不错", "正面"),
    ("虽然送晚了，但是饭菜很好吃", "正面"),
    ("除了贵没毛病", "正面"),

    # 负面评论
    ("难吃，再也不会买了", "负面"),
    ("太难吃了，完全不值得推荐", "负面"),
    ("送货太慢了，服务态度还很差", "负面"),
    ("味道很差，吃完就想吐", "负面"),
    ("以后绝对不会再买", "负面"),
    ("太贵了，完全不值这个价格", "负面"),
    ("真是太好了，又让我等了两个小时", "负面"),
    ("包装挺好，就是产品完全不能用", "负面"),
    ("除了便宜一无是处", "负面"),

    # 中性评论
    ("今天收到了商品，包装完好", "中性"),
    ("外卖员名字跟我们家名字一样", "中性"),
    ("这个在哪里买", "中性"),
    ("一般般，没有特别好也没有特别差", "中性"),
    ("买了一杯奶茶，价格十五元", "中性"),
    ("换了一个口味，还不知道好不好喝", "中性"),
]


correct_count = 0
wrong_cases = []

print("=" * 60)
print("SocialInsight 情感分类23条基线测试")
print("=" * 60)

for index, (comment, expected) in enumerate(test_cases, 1):
    result = analyze_sentiment(comment)

    actual = result["sentiment_label"]
    score = result["sentiment_score"]

    is_correct = actual == expected

    if is_correct:
        correct_count += 1
        status = "正确"
    else:
        status = "错误"
        wrong_cases.append({
            "comment": comment,
            "expected": expected,
            "actual": actual,
            "score": score
        })

    print(f"\n第{index}条：{comment}")
    print(f"预期：{expected} | 实际：{actual} | 分数：{score} | {status}")


total_count = len(test_cases)
accuracy = correct_count / total_count * 100

print("\n" + "=" * 60)
print("测试结果汇总")
print("=" * 60)

print(f"测试总数：{total_count}")
print(f"分类正确：{correct_count}")
print(f"分类错误：{total_count - correct_count}")
print(f"测试准确率：{accuracy:.2f}%")

if wrong_cases:
    print("\n错误样本：")
    for item in wrong_cases:
        print(f"\n评论：{item['comment']}")
        print(f"预期：{item['expected']}")
        print(f"实际：{item['actual']}")
        print(f"分数：{item['score']}")
else:
    print("\n所有测试样本均分类正确！")