import pandas as pd

# 저장된 CSV 파일 불러오기
df = pd.read_csv("contrib-loss-sum-result.csv")

# 그룹 나누기
normal_data = df[df['data_index'] < 50000]
shuffled_data = df[df['data_index'] >= 50000]

# 평균 기여도 계산
print(f"✅ 정상 데이터 평균 기여도: {normal_data['contribution_score'].mean():.6f}")
print(f"❌ 셔플 데이터 평균 기여도: {shuffled_data['contribution_score'].mean():.6f}")

# 셔플 데이터 중 기여도가 플러스인 것의 비율 (%)
plus_shuffled = (shuffled_data['contribution_score'] > 0).mean() * 100
print(f"⚠️ 셔플 데이터 중 기여도가 (+)인 비율: {plus_shuffled:.2f}%")