PATH = "contrib-loss-sum-result.csv"   # <- 너 파일명으로 바꿔

#self_influence_scores.txt
# loss-trajectory.txt
# contrib-loss-sum-result.csv
in_path = "self_influence_scores.txt" 
out_path = "self-influence.txt" 


with open(in_path, "r", encoding="utf-8", errors="ignore") as fin, \
     open(out_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        # 공백/탭 등 "어떤 whitespace든" 기준으로 split
        parts = line.split()
        # 맨 마지막 토큰만 값이라고 가정
        fout.write(parts[-1] + "\n")

print("Saved:", out_path)

