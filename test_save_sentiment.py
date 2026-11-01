from src.data_processor import load_csv, clean_dataframe
from src.sentiment_analyzer import (
    analyze_dataframe,
    save_sentiment_results
)
import pandas as pd
import os

# 1. 读取原始数据
df = load_csv("data/raw/test_comments.csv")

# 2. 清洗数据
cleaned_df, stats = clean_dataframe(df)

# 3. 情感分析
result_df = analyze_dataframe(cleaned_df)

# 4. 保存结果
output_path = save_sentiment_results(result_df)

# 5. 检查保存结果
saved_df = pd.read_csv(output_path, encoding="utf-8-sig")

print("\n========== 保存测试 ==========")
print("文件是否存在：", os.path.exists(output_path))
print("保存前数据条数：", len(result_df))
print("保存后数据条数：", len(saved_df))
print("情感分析字段：", 
      "sentiment_score" in saved_df.columns,
      "sentiment_label" in saved_df.columns)

assert len(result_df) == len(saved_df)
assert "sentiment_score" in saved_df.columns
assert "sentiment_label" in saved_df.columns

print("\n测试通过：情感分析结果保存成功！")
