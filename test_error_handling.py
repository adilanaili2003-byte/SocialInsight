from src.data_processor import load_csv, clean_dataframe

try:
    df = load_csv("data/raw/invalid_comments.csv")
    cleaned_df, stats = clean_dataframe(df)

except ValueError as e:
    print("成功捕获异常：", e)

