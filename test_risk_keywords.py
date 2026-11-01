from src.risk_analyzer import identify_risk_keywords

test_comments = [
    "这个产品质量太差了，用了几天就坏了",
    "物流太慢了，等了好几天才收到",
    "客服态度恶劣，根本不回复消息",
    "这个产品价格太贵了，性价比低",
    "申请退款困难，售后一直不处理",
    "这个产品真的很好用，非常满意"
]

print("========== 舆情风险关键词识别测试 ==========")

for comment in test_comments:
    categories = identify_risk_keywords(comment)

    print(f"\n评论：{comment}")
    print(f"风险类别：{categories}")

