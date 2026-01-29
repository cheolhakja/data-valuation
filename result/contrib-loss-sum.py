import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os, random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
g = torch.Generator()
g.manual_seed(42)

def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(x)

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        return x, y, idx


# 데이터 준비
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set_raw = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_set = IndexedDataset(train_set_raw)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


# 모델 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device) 
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion_mean = nn.CrossEntropyLoss() # 학습용
criterion_none = nn.CrossEntropyLoss(reduction='none') # 개별 Loss 측정용

# 기여도 저장용 (6만 개의 데이터셋에 대하여)
cumulative_contribution = torch.zeros(60000).to(device)

epochs = 10

epoch_pbar = tqdm(range(1, epochs + 1), desc="Total Epochs")

measure_loader = DataLoader(train_set, batch_size=100, shuffle=False)

for epoch in range(1, epochs + 1):
    model.eval()
    before_losses = torch.zeros(60000, device=device)

    with torch.no_grad():
        for imgs, labels, idxs in tqdm(measure_loader, desc=f"Epoch {epoch} [1/3] Measuring Before", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            idxs = idxs.to(device)
            loss = criterion_none(model(imgs), labels)
            before_losses[idxs] = loss

    model.train()
    running_loss = 0.0
    for imgs, labels, idxs in tqdm(train_loader, desc=f"Epoch {epoch} [2/3] Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion_mean(model(imgs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    after_losses = torch.zeros(60000, device=device)
    with torch.no_grad():
        for imgs, labels, idxs in tqdm(measure_loader, desc=f"Epoch {epoch} [3/3] Measuring After", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            idxs = idxs.to(device)
            loss = criterion_none(model(imgs), labels)
            after_losses[idxs] = loss

    cumulative_contribution += (before_losses - after_losses)
    print(f"Epoch {epoch} Done (Avg Loss: {running_loss/len(train_loader):.4f})")


import pandas as pd

# 1. 데이터 프레임 생성
df = pd.DataFrame({
    'data_index': range(60000),
    'contribution_score': cumulative_contribution.cpu().numpy()
})

# 2. CSV 파일로 저장
df.to_csv("contrib-loss-sum-result.csv", index=False)

print("인덱스가 포함된 결과가 contrib-loss-sum-result.csv에 저장되었습니다.")