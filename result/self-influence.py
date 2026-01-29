import torch
import torchvision
import torchvision.transforms as transforms
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
# 이미지를 텐서로 바꾸고 정규화
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f"학습 데이터 개수: {len(train_set)}")
print(f"테스트 데이터 개수: {len(test_set)}")

indices = list(range(60000))
mini_train_set = torch.utils.data.Subset(train_set, indices)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=32,
    shuffle=True,
    num_workers=0,          
    generator=g,
    worker_init_fn=seed_worker
)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

import torch
import torch.nn as nn
import torch.optim as optim
import os

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1) # Trace-In은 SGD에서 가장 잘 작동한다고 하네요

print(torch.cuda.is_available())

save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

epochs = 10
checkpoint_intervals = list(range(1, epochs + 1))

model.train()
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch} 완료")
    
    #체크포인트 저장
    if epoch in checkpoint_intervals:
        torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch}.pth")

print("학습 및 체크포인트 저장 완료!")
#===============================
# mini_train_set의 인덱스 순서대로 점수가 쌓인다고 함
tracin_scores = torch.zeros(len(mini_train_set))
learning_rate = 0.1

score_loader = torch.utils.data.DataLoader(mini_train_set, batch_size=1, shuffle=False)

for epoch in checkpoint_intervals:
    checkpoint_path = f"{save_dir}/model_epoch_{epoch}.pth"
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval() 
    print(f"에포크 {epoch} 체크포인트 계산 중...")
    
    for i, (image, label) in enumerate(score_loader):
        image, label = image.to(device), label.to(device)
        
        model.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        
        grad_norm_sq = 0.0
        for p in model.fc[-1].parameters():
            if p.grad is not None:
                grad_norm_sq += torch.norm(p.grad.data, p=2).item() ** 2
        
        tracin_scores[i] += learning_rate * grad_norm_sq

print("모든 데이터의 기여도 측정 완료!")


#===============================저장용===============================
import numpy as np



out_path2 = "self_influence_scores.txt"
idx = np.arange(len(tracin_scores))
np.savetxt(out_path2, np.column_stack([idx, tracin_scores]), fmt=["%d", "%.8f"])
print(f"Saved: {out_path2}")