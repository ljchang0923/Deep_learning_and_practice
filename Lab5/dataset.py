import torch
import os
import numpy as np
import csv
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2 as cv

default_transform = transforms.Compose([
    transforms.ToTensor()
    ])

def getlist(root, mode):
    seq_list, cond_list = [], []
    if mode == "train":
        record_path = os.listdir(root + "/train/")
        seq_list = [f"{root}/train/{record}/{i}/" for record in record_path for i in range(256)]
    elif mode == "validate":
        record_path = os.listdir(root + "/validate/")
        seq_list = [f"{root}/validate/{record}/{i}/" for record in record_path for i in range(256)]
    elif mode == "test":
        record_path = os.listdir(root + "/test/")
        seq_list = [f"{root}/test/{record}/{i}/" for record in record_path for i in range(256)]
    else:
        raise NameError("the mode doesn't exist")
    
    return seq_list

class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        
        self.root = args.data_root
        self.seq_list = getlist(self.root, mode)
        self.transform = transform
        self.seed_is_set = False
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def __len__(self):
        return len(self.seq_list)
        
    def get_seq(self, index):
        
        img_paths = [img_file for img_file in os.listdir(self.seq_list[index]) if img_file.endswith(".png")]
        # print("image path: ", img_paths)
        seq = []
        for img_name in img_paths:
            # print(self.seq_list[index] + img_name)
            img = cv.cvtColor(cv.imread(self.seq_list[index] + img_name), cv.COLOR_BGR2RGB)
            img = (img/255.0).astype(np.float32)
            img = self.transform(img).unsqueeze(0)
            seq.append(img)

        seq = torch.cat(seq, axis=0)
        return seq

    
    def get_csv(self, index):
        action = pd.read_csv(f"{self.seq_list[index]}actions.csv", header=None)
        position = pd.read_csv(f"{self.seq_list[index]}endeffector_positions.csv", header=None)
        cond = np.concatenate([action, position], axis=1)
        cond = torch.Tensor(cond)
        
        return cond
        
    
    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq(index)
        cond =  self.get_csv(index)
        return seq, cond
