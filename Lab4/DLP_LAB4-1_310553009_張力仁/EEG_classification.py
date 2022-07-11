import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from model import EEGNet, DeepConvNet
from dataloader import read_bci_data

BATCH_SIZE = 64

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloader(train_data, train_label, test_data, test_label):


    x_train = torch.Tensor(train_data)
    x_test = torch.Tensor(test_data)
    y_train = torch.Tensor(train_label).view(-1,1)
    y_test = torch.Tensor(test_label).view(-1,1)

    # 存成tensordataset
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    # 包成dataloader
    train_dl = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 0
    )

    test_dl = DataLoader(
        dataset = test_dataset,
        batch_size = len(test_dataset),
        shuffle = False,
        num_workers = 0
    )
    return train_dl, test_dl

def plot_result(info, record):

    plt.plot(record['val_acc'])
    plt.plot(record['train_acc'])
    plt.title(f'Accuracy({info["model"]}, {info["activation"]})')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(['val acc', 'train acc'])
    plt.savefig(f'log/{info["model"]}_{info["activation"]}_acc.png')
    plt.clf()

    plt.plot(record['val_loss'])
    plt.plot(record['train_loss'])
    plt.title(f'Testing Loss({info["model"]}, {info["activation"]})')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['val loss', 'train loss'])
    plt.savefig(f'log/{info["model"]}_{info["activation"]}_loss.png')
    plt.clf()

def train_model(model, train_dl, test_dl, config, device):

    optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=config['lr'], weight_decay=0.001)
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()

    #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    record ={'train_acc':[], 'train_loss':[], 'val_acc':[], 'val_loss':[]}
    total_loss = 0
    acc = 0
    max_acc = 0
    print('\n training start')
    model.train()
    for epoch in range(config['epoch']):
        for x_train, y_train in train_dl:
            model.train()
            optimizer.zero_grad()

            x_train, y_train = x_train.to(device), y_train.to(device).long().view(-1)
            pred = model(x_train)

            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().item()
            # acc += ((pred>0.5) == y_train).float().sum().detach().cpu().item()
            acc += (torch.max(pred, 1)[1] == y_train).float().sum().detach().cpu().item()

            optimizer.step()
            #lr_scheduler.step()

        val_loss, val_acc = val_model(model, test_dl, device)
        record["train_loss"].append(total_loss/len(train_dl))
        record["train_acc"].append(100*acc/len(train_dl.dataset))
        record["val_acc"].append(val_acc)
        record["val_loss"].append(val_loss)

        print(f"{epoch+1} epoch: train loss-> {total_loss/len(train_dl)}, acc-> {100*acc/len(train_dl.dataset)}")
        print(f"val loss-> {val_loss} acc-> {val_acc}")

        total_loss = 0
        acc = 0

        if(val_acc > max_acc):
            max_acc = val_acc
            torch.save(model.state_dict(), config['save_path'])
    torch.cuda.empty_cache()
    print("highedt acc: ", max_acc)
    return record

    
def val_model(model, test_dl, device):
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()

    val_loss = 0
    val_acc = 0
    model.eval()
    with torch.no_grad():
        for x_test, y_test in test_dl:
            x_test, y_test = x_test.to(device), y_test.to(device).long().view(-1)
            pred_val = model(x_test)
            val_loss = criterion(pred_val, y_test).detach().cpu().item()
            # val_acc = ((pred_val>0.5) == y_test).float().sum().detach().cpu().item()
            val_acc = (torch.max(pred_val, 1)[1] == y_test).sum().item()

    return val_loss/len(test_dl), 100 * val_acc / len(test_dl.dataset)


if __name__ =='__main__':
    device = get_device()
    # set_seed(34)
    mode = "train"
    print("get device: ", device)

    info = {
        "model": 'EEGNet', # EEGNet, DeepConvNet
        "activation": 'ReLU' # ELU, ReLU, LeakyReLU
    }

    train_data, train_label, test_data, test_label = read_bci_data()
    train_dl, test_dl = get_dataloader(train_data, train_label, test_data, test_label)
    
    if(info["model"]=="EEGNet"):
        model = EEGNet(info["activation"]).to(device)
    elif(info["model"]=="DeepConvNet"):
        kernel_num = [25, 50, 100, 200]
        model = DeepConvNet(info["activation"], kernel_num).to(device)
    else:
        raise BaseException("incorrect model")
    
    print(model)

    config = {
        'epoch': 450,
        'optimizer': 'Adam',
        'lr': 5e-3,
        "save_path": info["model"] + "_" + info["activation"]
    }
    if mode =='train':
        record = train_model(model, train_dl, test_dl, config, device)
    elif(mode=='val'):
        model.load_state_dict(torch.load('EEGNet_LeakyReLU_best'))
        val_loss, val_acc = val_model(model, test_dl, device)
        print(val_acc)
    # output = val_model(model, test_dl, device)
    # output = [p for p in output for x in p]
    plot_result(info, record)

