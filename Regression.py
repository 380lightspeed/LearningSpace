import os
from tqdm import tqdm


from Layers import CustomLinearLayer
from Loss import MSELoss

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Dataloader

############################################## Get dataset

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
X_train, X_test, y_train, y_test = torch.tensor(X_train), torch.tensor(X_test), torch.tensor(y_train), torch.tensor(y_test)
print(f"X.shape, y.shape - {X.shape, y.shape}")
print(f"X_train, X_test, y_train, y_test - {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")



############################################## Define Dataloader

class DataGenerator(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    
    def __len__(self):
        return y.shape[0]
    
    def __get_item_(self, idx):
        return X[idx], y



############################################## Define Model

class MLP(nn.Module):
    def __init__(self, input_dims:int, output_dims:int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dims)
        self.l1 = CustomLinearLayer(input_dims, 20)
        self.l2 = CustomLinearLayer(20,20)
        self.l3 = CustomLinearLayer(20,output_dims)
        self.activation = nn.GELU()
        self.seq = nn.Sequential(
                                    self.l1, self.activation,
                                    self.l2, self.activation,
                                    self.l3, self.activation)

        def foward(self, X):
            return self.seq(self.norm(X))
        

############################################## define run step

def run_step(model, dataloader, optimizer, device,training:bool, verbose=True):
    iterator = tqdm(dataloader) if verbose else dataloader

    for batch in dataloader:
        pass

input_dims = X_train.shape[1]
batch_size = 128
