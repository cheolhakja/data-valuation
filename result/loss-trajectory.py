import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class IndexedMNIST(Dataset):
    def __init__(self, train=True):
        self.mnist = datasets.MNIST(root='./data', train=train, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
    def __getitem__(self, index):
        data, target = self.mnist[index]
        return data, target, index  
    def __len__(self):
        return len(self.mnist)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples = 60000
num_epochs = 10
batch_size = 128

train_loader = DataLoader(IndexedMNIST(train=True), batch_size=batch_size, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self). __init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*13*13, 10)
        )
    def forward(self, x):
        return self.conv(x)

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(reduction="none") # 인스턴스 손실 계산용

aum_scores = torch.zeros(num_samples, device=device)

print(f"--- {device}에서 학습 및 점수 측정 시작 ---")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for x, y, idx in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        losses = criterion(logits, y)  # (batch_size,)
        
        loss_mean = losses.mean()
        loss_mean.backward()
        optimizer.step()

        aum_scores[idx] += losses.detach()
        epoch_loss += loss_mean.item()

    print(f"Epoch {epoch+1} - Average Loss: {epoch_loss/len(train_loader):.4f}")



print("--- 결과 저장 중... ---")


scores_cpu = aum_scores.cpu().numpy()

with open("loss-trajectory.txt", "w", encoding="utf-8") as f:
    for i, score in enumerate(scores_cpu):
        # 인덱스 : 누적 손실 형태 (소수점 4자리까지)
        f.write(f"{i} : {score:.4f}\n")

print("save complete")
