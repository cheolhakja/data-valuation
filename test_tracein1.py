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

#시드고정
# 
#     
# 이미지를 텐서로 바꾸고 정규화
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f"학습 데이터 개수: {len(train_set)}")
print(f"테스트 데이터 개수: {len(test_set)}")


train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=32,
    shuffle=True,
    num_workers=0,          
    generator=g,
    worker_init_fn=seed_worker
)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
#===============================

import torch
import torch.nn as nn
import torch.optim as optim
import os

# 모델 정의 
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

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

print("학습 시작...")
num_epochs = 10
history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # 에포크 끝날 때마다 가중치와 학습률 저장
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'lr': optimizer.param_groups[0]['lr']
    }, checkpoint_path)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f} -> 저장 완료")

print("모든 체크포인트가 저장되었습니다.")


#===============================
def get_gradient_vector(model, criterion, x, y):
    """특정 데이터 한 개에 대한 그래디언트를 계산하여 1차원 벡터로 반환"""
    model.zero_grad()
    
    # 배치 차원 추가 및 디바이스 이동
    x = x.unsqueeze(0).to(device)
    
    # 숫자(int)를 텐서로 변환하고 배치 차원 추가
    if isinstance(y, int):
        y = torch.tensor([y]).to(device) # 숫자를 리스트에 담아 텐서로 변환 [label]
    else:
        y = y.unsqueeze(0).to(device)
        
    output = model(x)
    loss = criterion(output, y)
    
    # 그래디언트 계산
    grads = torch.autograd.grad(loss, model.parameters())
    
    # 모든 레이어의 그래디언트를 하나로 합침
    return torch.cat([g.reshape(-1) for g in grads])

def calculate_trace_in(train_idx, test_idx, train_set, test_set, checkpoint_dir):
    """특정 학습 데이터와 테스트 데이터 사이의 Trace-In 점수 계산"""
    x_train, y_train = train_set[train_idx]
    x_test, y_test = test_set[test_idx]
    
    total_score = 0.0
    
    checkpoints = sorted(os.listdir(checkpoint_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    for ckpt_name in checkpoints:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        checkpoint = torch.load(ckpt_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        lr = checkpoint['lr']
        
        # 학습 데이터의 그래디언트
        grad_train = get_gradient_vector(model, criterion, x_train, y_train)
        # 테스트 데이터의 그래디언트
        grad_test = get_gradient_vector(model, criterion, x_test, y_test)
        
        # 내적 계산: lr * (grad_train · grad_test)
        dot_product = torch.dot(grad_train, grad_test)
        total_score += lr * dot_product.item()
        
    return total_score



from tqdm.auto import tqdm

def calculate_all_scores(mini_train_set, test_set, checkpoint_dir):
    
    total_scores = torch.zeros(len(mini_train_set)).to(device)
    
    checkpoints = sorted(os.listdir(checkpoint_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    for ckpt_name in tqdm(checkpoints, desc="Checkpoints"):
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        checkpoint = torch.load(ckpt_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        lr = checkpoint['lr']
        model.eval() # 연산 최적화

        for i in range(10000):  
            x_test, y_test = test_set[i]
            grad_test = get_gradient_vector(model, criterion, x_test, y_test)# 테스트 데이터 그래디언트 계산

        
            #모든 학습 데이터에 대해 내적 수행 
            for train_idx in range(len(mini_train_set)):
                x_train, y_train = mini_train_set[train_idx]
                grad_train = get_gradient_vector(model, criterion, x_train, y_train)
                
                dot_product = torch.dot(grad_train, grad_test)
                total_scores[train_idx] += lr * dot_product
            
    return total_scores.cpu().numpy()

# 실행
# all_scores = calculate_all_scores(mini_train_set, test_set, checkpoint_dir)

def calculate_all_scores_optimized(mini_train_set,test_set,checkpoint_dir,n_test=10000):
    total_scores = torch.zeros(len(mini_train_set), device=device)

    checkpoints = sorted(
        os.listdir(checkpoint_dir),
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    for ckpt_name in tqdm(checkpoints, desc="Checkpoints"):
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        lr = float(checkpoint.get('lr', 1.0))  
        model.eval()

        # 테스트 대표 gradient 만들기
        G_test = None
        for i in range(n_test):
            x_test, y_test = test_set[i]
            g = get_gradient_vector(model, criterion, x_test, y_test).detach()

            if G_test is None:
                G_test = g
            else:
                G_test += g

        # 각 학습 샘플 grad와 내적해서 누적
        for train_idx in range(len(mini_train_set)):
            x_train, y_train = mini_train_set[train_idx]
            g_train = get_gradient_vector(model, criterion, x_train, y_train).detach()

            total_scores[train_idx] += lr * torch.dot(g_train, G_test)

    return total_scores.detach().cpu().numpy()

#===============================마지막 레이어 + 내적 연산 최적화===============================
def get_lastlayer_params(model):
    last = model.fc[3] #모델이 바뀌면 이 부분도 바꿔야 함
    return [last.weight, last.bias]

def get_gradient_vector_lastlayer(model, criterion, x, y):
    """'마지막 레이어 파라미터'만 grad를 1D 벡터로 반환"""
    model.zero_grad()

    x = x.unsqueeze(0).to(device)
    if isinstance(y, int):
        y = torch.tensor([y], device=device)
    else:
        y = y.unsqueeze(0).to(device)

    out = model(x)
    loss = criterion(out, y)

    params = get_lastlayer_params(model)
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)

    return torch.cat([g.reshape(-1) for g in grads])

def calculate_all_scores_optimized_and_lastlayer(train_set,test_set,checkpoint_dir,n_test=10000):

    total_scores = torch.zeros(len(train_set), device=device)

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_') and f.endswith('.pth')]
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    for ckpt_name in tqdm(checkpoints, desc="Checkpoints"):
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        lr = float(checkpoint.get("lr", 1.0))
        model.eval()

        # 테스트 gradient 만들기
        G_test = None
        for i in range(n_test):
            x_t, y_t = test_set[i]
            g_t = get_gradient_vector_lastlayer(model, criterion, x_t, y_t).detach()

            if G_test is None:
                G_test = g_t
            else:
                G_test += g_t

        # 학습 데이터(last-layer만 사용) grad와 내적
        for train_idx in range(len(train_set)):
            x_tr, y_tr = train_set[train_idx]
            g_tr = get_gradient_vector_lastlayer(model, criterion, x_tr, y_tr).detach()

            total_scores[train_idx] += lr * torch.dot(g_tr, G_test)

    return total_scores.detach().cpu().numpy()


scores =calculate_all_scores_optimized_and_lastlayer(train_set, test_set, checkpoint_dir)


#===============================저장용===============================
import numpy as np

out_path = "tracin_scores.txt"
np.savetxt(out_path, scores, fmt="%.8f")
print(f"Saved: {out_path}")

out_path2 = "tracin_scores_with_index.txt"
idx = np.arange(len(scores))
np.savetxt(out_path2, np.column_stack([scores]), fmt=["%.8f"])
print(f"Saved: {out_path2}")

