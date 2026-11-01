from src.sentiment_analyzer import analyze_sentiment

test_comments = [
    "这个产品真的太好用了，非常满意！",
    "物流很快，包装也不错，值得推荐。",
    "产品还可以，没有特别惊喜。",
    "质量一般，感觉没有什么特别的。",
    "太差劲了，质量很差，非常失望！",
    "客服态度恶劣，完全不想再买了。"
]

print("========== 单条评论情感分析测试 ==========")

for comment in test_comments:
    result = analyze_sentiment(comment)

    print(f"\n评论：{comment}")
    print(f"情感分数：{result['sentiment_score']}")
    print(f"情感类别：{result['sentiment_label']}")
