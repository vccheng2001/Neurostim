# pytorch lstm for apnea detection
import numpy as np
import pandas as pd
import os
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split



#simple function which read the data from directory and return data and label
# you can make your own reader for other dataset.
def build_data(path):
    pos_path = os.path.join(path, "positive")
    neg_path = os.path.join(path, "negative")
    data, label = [], []

    pos_files = os.listdir(pos_path)
    neg_files = os.listdir(neg_path) 

    print('num pos: ',len(pos_files))
    print('num neg: ', len(neg_files))

    # load pos, neg files into data
    for file in pos_files:
        f = os.path.join(pos_path, file)
        data.append(np.loadtxt(f,delimiter="\n", dtype=np.float64))
        label.append(1)
    for file in neg_files:
        f = os.path.join(pos_path, file)
        data.append(np.loadtxt(f,delimiter="\n", dtype=np.float64))
        label.append(0)
    return data, label

# dataset definition
class ApneaDataset(Dataset):
    # load the dataset
    def __init__(self, root, dataset, apnea_type, excerpt):
        # load the csv file as a dataframe
        self.root = root
        self.dataset = dataset
        self.apnea_type = apnea_type
        self.excerpt = excerpt
        self.path = f"{root}{dataset}/{apnea_type}_{excerpt}"
        print('Extracting pos/neg sequences from: ', self.path)
        self.data, self.label = build_data(self.path)

    def __len__(self):
        return len(self.data)

    # get a row at an index
    def __getitem__(self, idx):
        seq = self.preprocess(self.data[idx])
        label = self.label[idx]
        return seq, label


    def preprocess(data):
        return data

    # get indexes for train and test rows
    def get_splits(self, test_frac=0.3):
        # determine sizes
        test_size = round(test_frac * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# prepare the data
if __name__ == "__main__":
    root= "data/"
    dataset = "dreams"
    apnea_type="osa"
    excerpt=1
    dataset = ApneaDataset(root,dataset,apnea_type,excerpt)
    train_dataloader = DataLoader(dataset=dataset,\
                                 batch_size=batch_size,\
                                 shuffle=True)
    test_data = ModelNet40Dataset(root=root, augment=True, full_dataset=full_dataset,  split='test')
