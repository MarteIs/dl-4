import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.optimizer import Optimizer

class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)

                beta1, beta2 = group['betas']
                lr = group['lr']
                weight_decay = group['weight_decay']
                momentum = state['momentum']

                update = beta1 * momentum + (1 - beta1) * grad
                update = torch.sign(update)
                
                momentum.mul_(beta2).add_(grad, alpha=1 - beta2)
                
                if weight_decay != 0:
                    update.add_(p, alpha=weight_decay)
                
                p.add_(update, alpha=-lr)

        return loss

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def create_synthetic_data(num_samples=1000, img_size=32):
    X = torch.randn(num_samples, 3, img_size, img_size)
    y = torch.randint(0, 10, (num_samples,))
    return X, y

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train, y_train = create_synthetic_data(5000)
    X_test, y_test = create_synthetic_data(1000)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(35):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        print(f'Epoch {epoch + 1}: Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%')

if __name__ == "__main__":
    train()


# Epoch 1: Loss: 2.3064 | Acc: 10.32%
# Epoch 2: Loss: 2.2990 | Acc: 10.52%
# Epoch 3: Loss: 2.2892 | Acc: 12.22%
# Epoch 4: Loss: 2.2786 | Acc: 12.72%
# Epoch 5: Loss: 2.2633 | Acc: 20.26%
# Epoch 6: Loss: 2.2435 | Acc: 21.96%
# Epoch 7: Loss: 2.2154 | Acc: 25.78%
# Epoch 8: Loss: 2.1886 | Acc: 32.80%
# Epoch 9: Loss: 2.1516 | Acc: 37.78%
# Epoch 10: Loss: 2.1102 | Acc: 42.36%
# Epoch 11: Loss: 2.0550 | Acc: 44.62%
# Epoch 12: Loss: 2.0026 | Acc: 45.26%
# Epoch 13: Loss: 1.9369 | Acc: 47.30%
# Epoch 14: Loss: 1.8749 | Acc: 48.58%
# Epoch 15: Loss: 1.8049 | Acc: 50.90%
# Epoch 16: Loss: 1.7316 | Acc: 52.82%
# Epoch 17: Loss: 1.6506 | Acc: 56.18%
# Epoch 18: Loss: 1.5703 | Acc: 57.68%
# Epoch 19: Loss: 1.4892 | Acc: 60.64%
# Epoch 20: Loss: 1.4011 | Acc: 63.28%
# Epoch 21: Loss: 1.3269 | Acc: 65.82%
# Epoch 22: Loss: 1.2499 | Acc: 68.96%
# Epoch 23: Loss: 1.1539 | Acc: 71.36%
# Epoch 24: Loss: 1.0752 | Acc: 73.14%
# Epoch 25: Loss: 0.9896 | Acc: 76.34%
# Epoch 26: Loss: 0.9084 | Acc: 78.72%
# Epoch 27: Loss: 0.8351 | Acc: 80.92%
# Epoch 28: Loss: 0.7615 | Acc: 83.10%
# Epoch 29: Loss: 0.6841 | Acc: 85.72%
# Epoch 30: Loss: 0.6231 | Acc: 87.62%
# Epoch 31: Loss: 0.5475 | Acc: 90.00%
# Epoch 32: Loss: 0.4773 | Acc: 92.22%
# Epoch 33: Loss: 0.4184 | Acc: 94.00%
# Epoch 34: Loss: 0.3602 | Acc: 95.50%
# Epoch 35: Loss: 0.3142 | Acc: 96.46%