import numpy as np
import torch 
import torch.nn as nn
from scipy import io
import os
from torch.utils.data import DataLoader, TensorDataset

class SCCNet(nn.Module):
    def __init__(self):
        super(SCCNet, self).__init__()
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 22, (22, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2   = nn.BatchNorm2d(20)
        # self.SquareLayer = square_layer()
        self.Drop1 = nn.Dropout(0.5)
        self.AvgPool1 = nn.AvgPool2d((1, 62), stride=(1, 12))
        self.classifier = nn.Linear(840, 4, bias=True)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        x = x ** 2
        x = self.Drop1(x)
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = x.view(-1, 840)
        x = self.classifier(x)

        #x = self.softmax(x)
        return x

subject = 1
device = 'cuda'
with open(f"fig_MI/sub{subject}_MI_data.npy", 'rb') as f:
    train_data = np.load(f)
with open(f'fig_MI/sub{subject}_MI_label.npy', 'rb') as f:
    train_label = np.load(f)

path = '/home/ubuntu/BCI/BCI_data'
subject_e = f"BCIC_S0{subject}_E.mat"
# subject_t = f"BCIC_S0{subject}_T.mat"
# train_data = io.loadmat(os.path.join(path, subject_t))['x_train']
# train_label = io.loadmat(os.path.join(path, subject_t))['y_train'].reshape(-1)

test_data = io.loadmat(os.path.join(path, subject_e))['x_test']
test_label = io.loadmat(os.path.join(path, subject_e))['y_test'].reshape(-1)

print(test_data.shape)
print(train_data.shape)

train_data = torch.Tensor(train_data).unsqueeze(1).to(device)
train_label = torch.Tensor(train_label).long().to(device)
test_data = torch.Tensor(test_data).unsqueeze(1).to(device)
test_label = torch.Tensor(test_label).long().to(device)

train_dataset = TensorDataset(train_data, train_label)
test_dataset = TensorDataset(test_data, test_label)
train_dl = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)
test_dl = DataLoader(
    dataset=test_dataset,
    batch_size=len(test_dataset),
    shuffle=False,
    num_workers=0
)

model = SCCNet().to(device)
epochs = 500
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()
max_acc = 0
for epoch in range(epochs):
    total_loss = 0
    for x, y in train_dl:

        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    with torch.no_grad():
        for x_test, y_test in test_dl:
            x_test, y_test = x_test.to(device), y_test.to(device)
            pred_test = model(x_test)
            pred_test = torch.argmax(pred_test, 1)
            acc = (pred_test==y_test).sum().item()
        
    acc /= len(test_dl.dataset)

    if max_acc < acc:
        max_acc = acc
        torch.save(model.state_dict(), f'fig_MI/sub{subject}_best_model.pth')

    
    print("epoch: %d || loss: %.5f || testing acc: %.4f"%(epoch, total_loss/len(train_dl), acc))

print(max_acc)
