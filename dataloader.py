# pytorch lstm for apnea detection
import numpy as np
import pandas as pd
import os
import sys
import random
import torch 
from torch.utils.data import Dataset
from torch.utils.data import random_split

'''Split dataset into train/test in preparation for apnea detection model'''

class ApneaDataset(Dataset):
    # load the dataset
    def __init__(self, root, dataset, apnea_type, excerpt):
        # load the csv file as a dataframe
        self.root = root
        self.dataset = dataset
        self.apnea_type = apnea_type
        self.excerpt = excerpt
        self.timesteps = None
        self.path = f"{root}{dataset}/postprocessing/excerpt{excerpt}"
        self.data, self.label, self.files = self.build_data(self.path)

    def __len__(self):
        return len(self.data)

    ''' retrieve one sample '''
    def __getitem__(self, idx):
        seq = self.preprocess(self.data[idx])
        label = self.label[idx]
        file = self.files[idx]
        return seq, label, file

    ''' make sure data is the correct number of timesteps '''
    def preprocess(self, data):
        data = data[:self.timesteps]
        return data

    ''' split data into train, test'''
    def get_splits(self, test_frac=0.3):
        # determine sizes
        test_size = round(test_frac * len(self.data))
        train_size = len(self.data) - test_size
        # calculate the split
        print(f"train size: {train_size}, test_size: {test_size}")
        return random_split(self, [train_size, test_size])

   
    ''' build pos, neg files '''
    def build_data(self, path):
        print(f'Extracting pos/neg sequences from: {self.path}')

        pos_path = os.path.join(path, "positive")
        neg_path = os.path.join(path, "negative")
        data, label, files = [], [], []

        pos_files = os.listdir(pos_path)
        neg_files = os.listdir(neg_path) 

        # check timesteps
        self.timesteps = len(pos_files[0])

        # store file labels 
        map = {}
        for file in pos_files:
            map[file] = 1
        for file in neg_files:
            map[file] = 0

        print(f'Number of positive files: {len(pos_files)}')
        print(f'Number of negative files: {len(neg_files)}')


        # Randomly shuffle files 
        all_files = pos_files + neg_files
        random.shuffle(all_files)

        # Separate into positive, negative datasets
        for file in all_files:
            if map[file] == 1:
                f = os.path.join(pos_path, file)
            else:
                f = os.path.join(neg_path, file)
            arr = np.loadtxt(f,delimiter="\n", dtype=np.float64)
            data.append(np.expand_dims(arr,-1))
            label.append(map[file])
            files.append(file)


        return data, label, files

# prepare the data
if __name__ == "__main__":
    pass
    # root= "data/"
    # dataset = "dreams"
    # apnea_type="osa"
    # batch_size=128
    # excerpt=1
    # dataset = ApneaDataset(root,dataset,apnea_type,excerpt)
    # train_loader = DataLoader(dataset=dataset,\
    #                              batch_size=batch_size,\
    #                              shuffle=True)
    # seq, label, file = iter(train_loader).next()
    # print('seq: ', seq.shape)
    # print('label: ', label)
    
    # print('file', file.shape)
