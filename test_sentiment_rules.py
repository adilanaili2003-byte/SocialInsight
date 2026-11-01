from src.sentiment_refiner import apply_rule_overrides

test_cases = [
    # 正面
    ("味道很好喝，包装也不错", "正面"),
    ("虽然送晚了，但是饭菜很好吃", "正面"),
    ("包装有点破损，不过味道很好", "正面"),

    # 负面
    ("今天收到了商品，但是质量很差", "负面"),
    ("商品包装完好，但是完全不能用", "负面"),
    ("不是很好吃", "负面"),
    ("真是太好了，又让我等了两个小时", "负面"),

    # 中性
    ("今天收到了商品，包装完好", "中性"),
    ("外卖员名字跟我们家名字一样", "中性"),
    ("这个在哪里买", "中性"),
    ("买了一杯奶茶，价格十五元", "中性"),
]

print("=" * 55)
print("情感规则冲突测试")
print("=" * 55)

correct = 0

for text, expected in test_cases:
    actual = apply_rule_overrides(text)

    status = "正确" if actual == expected else "错误"

    if actual == expected:
        correct += 1

    print(f"\n评论：{text}")
    print(f"预期：{expected} | 实际：{actual} | {status}")

print("\n" + "=" * 55)
print(f"测试结果：{correct}/{len(test_cases)}")
print(f"准确率：{correct / len(test_cases) * 100:.2f}%")