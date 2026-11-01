from src.agents.workflow import run_analysis


print("\n========== SocialInsight 端到端测试 ==========")

result = run_analysis(
    file="data/raw/test_comments.csv",
    generate_ai_report=True,
    save_results=True
)


print("\n========== 数据清洗结果 ==========")

print(result["cleaning_stats"])


print("\n========== 舆情风险结果 ==========")

risk_result = result["risk_result"]

print("总评论数：", risk_result["total_count"])
print("负面评论数：", risk_result["negative_count"])
print("负面比例：", risk_result["negative_ratio"])
print("风险等级：", risk_result["risk_level"])


print("\n========== 大模型报告 ==========")

print(result["report"])


print("\n========== 端到端测试结果 ==========")

assert result["report"] is not None
assert len(result["report"]) > 0

print("端到端测试通过！")
print("分析结果和报告均已保存。")