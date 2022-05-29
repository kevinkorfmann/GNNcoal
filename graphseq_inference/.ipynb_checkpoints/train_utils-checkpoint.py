import random
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import os
from tqdm import tqdm
import numpy as np


class MyDataset():
    
    def __init__(self, file_names):
        self.file_names = file_names
        
    def __getitem__(self, key):
        signal, mask = torch.load(open(self.file_names[key], "rb"))
        return signal, mask
    
    def __len__(self):
        return len(self.file_names)
    
    def shuffle(self):
        
        random.shuffle(self.file_names)
        


def get_train_test_dataset(data:str = "../dataset/", split:float = 0.9):

    file_names = os.listdir(data)#[0:1000]
    file_names = sorted(file_names)
    file_names = [data + file for file in file_names]
    train_split = int(len(file_names) * split)
    train_filenames = file_names[:train_split]
    test_filenames = file_names[train_split:]
    train_dataset = MyDataset(train_filenames)
    test_dataset = MyDataset(test_filenames)
    
    return train_dataset, test_dataset

def get_train_test_datasets(data_list:list[str] = ["../dataset_0/", "../dataset_100/"], split:float = 0.9):
    """ Takes a list of directories and produces train and test datasets
    """

    all_file_names = []
    for data in data_list:
        
        file_names = os.listdir(data)#[0:1000]
        all_file_names += [data + file for file in file_names]
        
    file_names = all_file_names
    file_names = sorted(file_names)
    
    #file_names = [data + file for file in file_names]
    train_split = int(len(file_names) * split)
    train_filenames = file_names[:train_split]
    test_filenames = file_names[train_split:]
    train_dataset = MyDataset(train_filenames)
    test_dataset = MyDataset(test_filenames)
    
    return train_dataset, test_dataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        
def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))
