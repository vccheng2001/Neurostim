# pytorch lstm for apnea detection
import numpy as np
import pandas as pd
import os
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

timesteps = {'dreams': 120,
             'mit': 150,
             'dublin': 160}


# dataset definition
class ApneaDataset(Dataset):
    # load the dataset
    def __init__(self, root, dataset, apnea_type, excerpt):
        # load the csv file as a dataframe
        self.root = root
        self.dataset = dataset
        self.apnea_type = apnea_type
        self.excerpt = excerpt
        self.timesteps = timesteps[dataset]


        self.path = f"{root}{dataset}/{apnea_type}_{excerpt}"
        print('Extracting pos/neg sequences from: ', self.path)
        self.data, self.label, self.files = self.build_data(self.path)

    def __len__(self):
        return len(self.data)

    # get a row at an index
    def __getitem__(self, idx):
        seq = self.preprocess(self.data[idx])
        label = self.label[idx]
        file = self.files[idx]
        return seq, label, file


    def preprocess(self, data):
        data = data[:self.timesteps]
        return data

    # get indexes for train and test rows
    def get_splits(self, test_frac=0.3):
        # determine sizes
        test_size = round(test_frac * len(self.data))
        train_size = len(self.data) - test_size
        # calculate the split
        print(f"train size: {train_size}, test_size: {test_size}")
        return random_split(self, [train_size, test_size])

   
    def build_data(self, path):
        pos_path = os.path.join(path, "positive")
        neg_path = os.path.join(path, "negative")
        data, label, files = [], [], []

        pos_files = os.listdir(pos_path)
        neg_files = os.listdir(neg_path) 

        print('num pos: ',len(pos_files))
        print('num neg: ', len(neg_files))

        # load pos, neg files into data
        for file in pos_files:
            f = os.path.join(pos_path, file)
            arr = np.loadtxt(f,delimiter="\n", dtype=np.float64)
            if arr.shape[0] >= self.timesteps:
                data.append(np.expand_dims(arr,-1))
                label.append(1)
                files.append(file)
        for file in neg_files:
            f = os.path.join(neg_path, file)
            arr = np.loadtxt(f,delimiter="\n", dtype=np.float64)
            if arr.shape[0] >= self.timesteps:
                data.append(np.expand_dims(arr,-1))
                label.append(0)
                files.append(file)

        return data, label, files

# prepare the data
if __name__ == "__main__":
    root= "data/"
    dataset = "dreams"
    apnea_type="osa"
    batch_size=128
    excerpt=1
    dataset = ApneaDataset(root,dataset,apnea_type,excerpt)
    train_loader = DataLoader(dataset=dataset,\
                                 batch_size=batch_size,\
                                 shuffle=True)
    seq, label, file = iter(train_loader).next()
    print('seq: ', seq.shape)
    print('label: ', label)
    # print('file', file.shape)