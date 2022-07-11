from dataloader import RetinopathyLoader
import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import torchvision.models as models

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BATCH_SIZE = 8

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def i_am_handsome():
    pass

def train_model(model, train_dl, test_dl, config, device):
    optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr = config['lr'],
                momentum=config['momentum'], weight_decay=config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    record = {
        'train acc':[],
        'train loss':[],
        'val acc':[],
        'val loss':[],
    }
    max_acc = 0
    train_loss = 0
    correct = 0
    i=0
    for epoch in range(config['epoch']):
        model.train()
        for x_train, y_train in train_dl:
            optimizer.zero_grad()

            x_train, y_train = x_train.to(device), y_train.to(device).long().view(-1)
            pred = model(x_train)
            # print("label size: ",y_train.size())
            # print("pred size: ", pred.size())
            loss = criterion(pred, y_train)
            train_loss += loss.detach().cpu().item()
            correct += (torch.argmax(pred, dim=1) == y_train).float().sum().detach().cpu().item()
            loss.backward()
            optimizer.step()
            # i+=1
            # print(f"iteration {i}")
        
        val_acc, val_loss = val_model(model, test_dl, config, device) 
        train_loss = train_loss/len(train_dl)
        train_acc = correct/len(train_dl.dataset)
        print(f"{epoch+1} epoch: train loss->{train_loss} train_acc->{train_acc} val loss->{val_loss} val acc->{val_acc}")
        record['train loss'].append(train_loss)
        record['val loss'].append(val_loss)
        record['val acc'].append(val_acc)
        record['train acc'].append(train_acc)
        if(val_acc > max_acc):
            max_acc = val_acc
            torch.save(model.state_dict(), config['save_path'])

        with open("ResNet50_pre_tune_aug_log", 'a') as f:
            f.writelines(f"{epoch+1} epoch:\t{train_loss}\t{train_acc}\t{val_loss}\t{val_acc}\n")

        train_loss = 0
        correct = 0
    torch.cuda.empty_cache()
    return record



def val_model(model, test_dl, config, device):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    loss_val, correct_val = 0, 0
    with torch.no_grad():
        for x_test, y_test in test_dl:
            x_test, y_test = x_test.to(device), y_test.to(device).long().view(-1)

            pred = model(x_test)
            loss = criterion(pred, y_test)
            loss_val += loss.detach().cpu().item()
            correct_val += (torch.argmax(pred, dim=1) == y_test).float().sum().detach().cpu().item()

    return correct_val/len(test_dl.dataset), loss_val/len(test_dl)

def test_model(model, test_dl, device):

    model.eval()
    output = []
    correct = 0
    with torch.no_grad():
        for x_test, y_test in test_dl:
            x_test, y_test = x_test.to(device), y_test.to(device).long().view(-1)

            pred = model(x_test)
            output.append(torch.argmax(pred, dim=1))
            correct += (torch.argmax(pred, dim=1) == y_test).float().sum().detach().cpu().item()

    print(f"testing accuracy: {correct/len(test_dl.dataset)*100}%")
    return output

def plot_result(info, record):

    plt.plot(record['val acc'])
    plt.plot(record['train acc'])
    plt.title(f'Accuracy({info["model"]}, {info["train mode"]})')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(['val acc', 'train acc'])
    plt.savefig(f'log/{info["model"]}_{info["train mode"]}_acc.png')
    plt.clf()

    plt.plot(record['val loss'])
    plt.plot(record['train loss'])
    plt.title(f'Testing Loss({info["model"]}, {info["train mode"]})')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['val loss', 'train loss'])
    plt.savefig(f'log/{info["model"]}_{info["train mode"]}_loss.png')
    plt.clf()

if __name__ =="__main__":

    device = get_device()
    print("get device: ", device)
    mode = 'test'

    root_path = "data/"
    train_transform = trans.Compose([
        trans.ToTensor(),
        trans.RandomHorizontalFlip(0.5),
        trans.RandomVerticalFlip(0.5),
        trans.Normalize(mean = [0.37491241, 0.26017534, 0.18566248], std = [0.25253579, 0.17794982, 0.12907731])
        
    ])
    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize(mean = [0.37491241, 0.26017534, 0.18566248], std = [0.25253579, 0.17794982, 0.12907731])
        
    ])

    train_set = RetinopathyLoader(root_path, "train", train_transform)
    test_set = RetinopathyLoader(root_path, "test", test_transform)

    train_dl = DataLoader(dataset=train_set, batch_size = BATCH_SIZE, shuffle=True, num_workers=0)
    test_dl = DataLoader(dataset=test_set, batch_size = BATCH_SIZE, shuffle=False, num_workers=0)

    if mode == 'test':
        model = models.resnet18(pretrained=False)
        num_out = model.fc.in_features
        model.fc = nn.Linear(num_out, 5)
        # model = model.ResNet50(3,5).to(device)
        model.load_state_dict(torch.load('model/ResNet18_pretrained_tune_aug.pt'))
        model.to(device)
        output = test_model(model ,test_dl, device)
        output = [t.detach().cpu().item() for b in output for t in b]
        output = np.array(output)
        y_test = np.squeeze(pd.read_csv('test_label.csv'))

        print("size of y_test: ", y_test.shape)
        print("size of output: ", output)
        cm = confusion_matrix(y_test, output, normalize = 'pred')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig('confusion_mat_res18_pre.png')

    else:

        model = model.ResNet18(3,5).to(device)
        # model = model.ResNet50(3,5).to(device)

        ## with pretrained weight
        # model = models.resnet18(pretrained=True)
        # model = models.resnet50(pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = True
        
        # num_out = model.fc.in_features
        # model.fc = nn.Linear(num_out, 5)

        model.to(device)

        info = {
            "model": "ResNet18",
            "train mode": "pretrained_tune_aug" 
        }

        config = {
            'epoch': 10,
            'optimizer': 'SGD',
            'lr': 1e-3,
            'momentum':0.9,
            'weight_decay':5e-4,
            'save_path': f'model/{info["model"]}_{info["train mode"]}.pt'
        }
        print("start training")
        record = train_model(model, train_dl, test_dl, config, device)

        plot_result(info, record)
