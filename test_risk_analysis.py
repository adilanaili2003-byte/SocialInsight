import pandas as pd
from src.risk_analyzer import analyze_risks

# 构造测试数据
test_data = {
    "content": [
        "这个产品质量太差了，用了几天就坏了",
        "物流太慢了，等了好几天才收到",
        "客服态度恶劣，根本不回复消息",
        "这个产品价格太贵了，性价比低",
        "申请退款困难，售后一直不处理",
        "这个产品真的很好用，非常满意",
        "物流速度很快，包装也不错",
        "产品质量太差了，非常失望",
        "客服服务差，态度恶劣"
    ],
    "sentiment_label": [
        "负面",
        "负面",
        "负面",
        "负面",
        "负面",
        "正面",
        "正面",
        "负面",
        "负面"
    ],
    "topic_id": [
        0, 1, 2, 3, 4, 0, 1, 0, 2
    ]
}

df = pd.DataFrame(test_data)

# 执行风险分析
result = analyze_risks(df)

print("========== 舆情风险分析测试 ==========")

print(f"\n总评论数：{result['total_count']}")
print(f"负面评论数：{result['negative_count']}")
print(f"负面评论占比：{result['negative_ratio']:.2%}")
print(f"风险等级：{result['risk_level']}")

print("\n========== 风险类别统计 ==========")
for category, count in result["risk_counts"].items():
    print(f"{category}：{count}")

print("\n========== 负面主题分布 ==========")
for topic, count in result["topic_distribution"].items():
    print(f"主题 {topic}：{count} 条")

print("\n========== 负面评论明细 ==========")
print(result["negative_comments"][["content", "risk_categories"]])

