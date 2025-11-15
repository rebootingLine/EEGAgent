import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix
import time
import os

import torch.nn.init as init

def init_weights_health(m):
    if isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

    elif isinstance(m, nn.LayerNorm):
        pass


class healthMDD(nn.Module):
    def __init__(self, n_channels=19, n_samples=1280, hidden=64, p=0.):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 24, 75, 1),
            nn.LayerNorm(1206),
            nn.ReLU(),
            nn.AvgPool1d(5, 5),
            nn.Dropout(p),
            nn.Conv1d(24, 48, 15, 1),
            nn.LayerNorm(227),
            nn.ReLU(),
            nn.AvgPool1d(5, 5),
            nn.Dropout(p),
            nn.Conv1d(48, hidden, 5, 1),
            nn.LayerNorm(41),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        x = self.conv(x).flatten(1) # (32, 48, 49)
        return self.fc(x)

def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        time0 = time.time()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            correct += (outputs.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.append(outputs.argmax(1).detach().cpu())
            all_labels.append(y_batch.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        time1 = time.time()
        print(f"Epoch {epoch+1}/{epochs}:{time1-time0:.2f}s - Loss: {total_loss/total:.4f} - Acc: {correct/total:.4f} - MF1: {macro_f1:.4f}")

        test_model(model, test_loader, device=device)


best_f1 = 0

def test_model(model, test_loader, device='cpu', class_names=None):
    global best_f1  
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(1)
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Test Accuracy: {acc:.4f}  |  Macro-F1: {macro_f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    if macro_f1 > 0.84 and macro_f1 > best_f1:
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "./checkpoints/best_model.pth")
        print(f"Model saved! Acc={acc:.4f}, Macro-F1={macro_f1:.4f}")
        best_f1 = macro_f1

    return acc, macro_f1

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    X_train, y_train = torch.load("./data/train.pt", weights_only=False)
    X_test, y_test = torch.load("./data/test.pt", weights_only=False)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print(X_train.shape)


    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = healthMDD(n_channels=X_train.shape[1], n_samples=X_train.shape[2])
    model.apply(init_weights_health)
    train_model(model, train_loader, test_loader, epochs=20, lr=1e-4, device='cuda')

    # 93 94