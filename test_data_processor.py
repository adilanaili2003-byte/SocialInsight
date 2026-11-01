import pandas as pd
from src.data_processor import clean_dataframe

# 读取测试数据
df = pd.read_csv("data/raw/test_comments.csv")

print("========== 原始数据 ==========")
print(df)
print("原始数据量：", len(df))

# 执行数据清洗
cleaned_df, stats = clean_dataframe(df)

print("\n========== 清洗后的数据 ==========")
print(cleaned_df)

print("\n========== 清洗统计结果 ==========")
for key, value in stats.items():
    print(f"{key}: {value}")
