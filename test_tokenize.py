from src.topic_analyzer import tokenize_text

test_comments = [
    "物流速度很快，包装也不错",
    "产品质量太差了，非常失望",
    "客服态度很好，解决问题很及时",
    "这个产品真的很好用"
]

print("========== 中文分词测试 ==========")

for comment in test_comments:
    words = tokenize_text(comment)

    print(f"\n原始评论：{comment}")
    print(f"分词结果：{words}")

