from src.data_processor import load_csv, clean_dataframe

# 读取 CSV 文件
df = load_csv("data/raw/test_comments.csv")

print("========== CSV 读取结果 ==========")
print(df.head())
print(f"\n成功读取 {len(df)} 条评论")

# 执行数据清洗
cleaned_df, stats = clean_dataframe(df)

print("\n========== 数据清洗统计 ==========")
for key, value in stats.items():
    print(f"{key}: {value}")

