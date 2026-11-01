from src.data_processor import load_csv, clean_dataframe, save_processed_data
import pandas as pd
import os

# 1. 读取原始数据
df = load_csv("data/raw/test_comments.csv")

# 2. 清洗数据
cleaned_df, stats = clean_dataframe(df)

# 3. 保存清洗结果
output_path = save_processed_data(cleaned_df)

# 4. 检查文件是否存在
print("\n========== 保存测试结果 ==========")
print("文件是否存在：", os.path.exists(output_path))

# 5. 重新读取保存的文件
saved_df = pd.read_csv(output_path, encoding="utf-8-sig")

print("保存前数据条数：", len(cleaned_df))
print("保存后数据条数：", len(saved_df))
print("\n保存的数据：")
print(saved_df)

# 6. 检查数据是否一致
assert len(cleaned_df) == len(saved_df)
assert list(cleaned_df["content"]) == list(saved_df["content"])

print("\n测试通过：清洗后的数据已正确保存！")
