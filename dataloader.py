# pytorch lstm for apnea detection
import numpy as np
import pandas as pd
import os
import sys
import random
import torch 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import shutil

'''Split dataset into train/test in preparation for apnea detection model'''

class ApneaDataset(Dataset):
    # load the dataset
    def __init__(self, root, dataset, apnea_type, excerpt):
        # load the csv file as a dataframe
        print('root', root)
        print(os.getcwd())
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
        seq = (self.data[idx])[:self.timesteps]
        label = self.label[idx]
        file = self.files[idx]
        return seq, label, file


    ''' split data into train, test'''
    def get_splits(self, test_frac=0.2):
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

        # get number of timesteps
        first_pos_file = os.path.join(pos_path, pos_files[0])
        self.timesteps = len(open(first_pos_file, 'r').readlines())
        print('Timesteps:', self.timesteps)
        # store file labels 
        map = {}
        for file in pos_files:
            map[file] = 1
        for file in neg_files:
            map[file] = 0

        '''' Downsampling if needed, for class balancing '''
        num_pos_files = len(pos_files)
        num_neg_files = len(neg_files)

       
        # Downsampling 
        if num_pos_files > num_neg_files * 2:
            print('Downsampling pos files')
            pos_files = random.sample(pos_files, num_neg_files * 2)
        if num_neg_files  > num_pos_files * 2:
            print('Downsampling neg files')
            neg_files = random.sample(neg_files, num_pos_files * 2)




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
            if arr.shape[0] >=  self.timesteps:
                arr = arr[:self.timesteps]
                data.append(np.expand_dims(arr,-1))
                label.append(map[file])
                files.append(file)



        print(f'Number of positive files: {len(pos_files)}')
        print(f'Number of negative files: {len(neg_files)}')

        return data, label, files

class ApneaDataloader(DataLoader):
    def __init__(self, root, dataset, apnea_type, excerpt, batch_size):
        self.dataset = ApneaDataset(root, dataset, apnea_type, excerpt)
        self.test_frac = 0.2
        self.batch_size = batch_size
        self.train_data, self.val_data = self.dataset.get_splits(self.test_frac)

    def get_data(self):
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        return self.train_loader, self.val_loader




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

''' Helper function to create directory '''
def init_dir(path): 
    if os.path.isdir(path): shutil.rmtree(path)
    if not os.path.isdir(path):
        os.makedirs(path)