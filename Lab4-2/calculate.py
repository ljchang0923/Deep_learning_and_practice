import cv2 as cv
import numpy as np
import pandas as pd

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

x_train, _ = getData("test")
root = "data/"
mean=np.array([0,0,0], dtype=float)
std=np.array([0,0,0], dtype=float)
for idx in range(len(x_train)):
    path = root + x_train[idx] + '.jpeg'
    img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
    img = img.transpose(2,0,1)
    img = img/255.0
    mean += np.mean(img, axis=(1,2))
    std += np.std(img, axis=(1,2))

print("mean: ", mean/len(x_train))
print("std: ", std/len(x_train))

# train mean = [0.37491241 0.26017534 0.18566248]
# train std = [0.25253579 0.17794982 0.12907731]
# test mean = [0.37695431 0.26036074 0.18607829]
# test std = [0.25360558 0.17819041 0.12943836]