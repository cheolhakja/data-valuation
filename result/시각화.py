import numpy as np
import matplotlib.pyplot as plt

# ====== 설정 ======
PATH = "tracein.txt"   # <- 너 파일명으로 바꿔

#self-influence.txt
# loss-trajectory.txt
# tracein.txt
START = 0
N = 1000
# ==================

def load_scores(path: str) -> np.ndarray:
    # 1) txt/csv "한 줄에 숫자 하나"인 경우
    try:
        arr = np.loadtxt(path, dtype=float)
        return arr
    except Exception:
        pass

    # 2) csv에 헤더가 있고 한 컬럼인 경우 (예: contribution_score)
    import pandas as pd
    df = pd.read_csv(path)
    # 첫 번째 숫자 컬럼을 자동으로 선택
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) == 0:
        raise ValueError("CSV에서 숫자 컬럼을 찾지 못했습니다.")
    return df[num_cols[0]].to_numpy(dtype=float)

scores = load_scores(PATH)

slice_scores = scores[START:START+N]
x = np.arange(START, START + len(slice_scores))

plt.figure()
plt.plot(x, slice_scores)
plt.xlabel("index")
plt.ylabel("contribution")
plt.title(f"{PATH} [{START}..{START+N-1}]")
plt.grid(True)
plt.show()
