import numpy as np
import ot
from ot.lp import free_support_barycenter  
import matplotlib.pyplot as plt

# 가상 데이터 생성 
np.random.seed(42)
client1 = np.random.normal(loc=[2, 2], scale=0.5, size=(30, 2))
client2 = np.random.normal(loc=[2.5, 2.5], scale=0.5, size=(30, 2))
client3 = np.random.normal(loc=[8, 8], scale=1.0, size=(30, 2)) # 노이즈 데이터

measures_locations = [client1, client2, client3]
measures_weights = [np.ones(len(c)) / len(c) for c in measures_locations]

# Wasserstein Barycenter 계산
k = 20 # 바리센터를 구성할 포인트 개수
X_init = np.random.normal(loc=[4, 4], scale=1, size=(k, 2))

barycenter_coords = free_support_barycenter(measures_locations, measures_weights, X_init)

print("--- 기준점 생성 완료 ---")

# 클라이언트별 기여도 계산 (바리센터와의 거리)
dists = []
for i, client in enumerate(measures_locations):
    M = ot.dist(client, barycenter_coords)
    
    d = ot.emd2(measures_weights[i], np.ones(k)/k, M)
    dists.append(d)
    print(f"클라이언트 {i+1} - 기준점과의 거리: {d:.4f}")

# 시각화
plt.figure(figsize=(10, 7))
plt.scatter(client1[:, 0], client1[:, 1], label='Client 1 (Good)', alpha=0.4)
plt.scatter(client2[:, 0], client2[:, 1], label='Client 2 (Good)', alpha=0.4)
plt.scatter(client3[:, 0], client3[:, 1], label='Client 3 (Noisy)', alpha=0.4)
plt.scatter(barycenter_coords[:, 0], barycenter_coords[:, 1], 
            label='FedBarycenter', c='black', marker='X', s=150, edgecolors='white')
plt.title("FedBary: Data Valuation using ot.lp.free_support_barycenter")
plt.legend()
plt.show()