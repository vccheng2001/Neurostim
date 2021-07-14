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
import wandb

'''Split dataset into train/test in preparation for apnea detection model'''

class ApneaDataset(Dataset):
    # load the dataset
    def __init__(self, cfg):
        # load the csv file as a dataframe
        self.pos_dir = cfg.positive_dir
        self.neg_dir = cfg.negative_dir
        self.root_dir = cfg.root_dir
        self.logger = cfg.logger
        self.data_root = os.path.join(self.root_dir, "data/")
        self.dataset = cfg.dataset
        self.apnea_type = cfg.apnea_type
        self.excerpt = cfg.excerpt
        self.test_frac = float(cfg.test_frac)
        self.timesteps = cfg.sample_rate * (int(cfg.seconds_before_apnea) + int(cfg.seconds_after_apnea))
        self.path = f"{self.data_root}{cfg.dataset}/postprocessing/excerpt{cfg.excerpt}"
        self.data, self.label, self.files = self.build_data(self.path)

    def __len__(self):
        return len(self.data)

    
    ''' retrieve one sample '''
    def __getitem__(self, idx):
        seq = self.data[idx]
        label = self.label[idx]
        file = self.files[idx]
        return seq, label, file


    ''' split data into train, test'''
    def get_splits(self, test_frac):
        # determine sizes
        test_size = round(float(test_frac) * len(self.data))
        train_size = len(self.data) - test_size
        # calculate the split
        print(f"train size: {train_size}, test_size: {test_size}")
        return random_split(self, [train_size, test_size])

   
    ''' build pos, neg files '''
    def build_data(self, path):
        print(f'Extracting pos/neg sequences from: {self.path}')

        if self.pos_dir:
            pos_path = self.pos_dir
        else:
            pos_path = os.path.join(path, "positive")

        if self.neg_dir:
            neg_path = self.neg_dir
        else:
            neg_path = os.path.join(path, "negative")
        data, label, files = [], [], []

    
        pos_files = os.listdir(pos_path)
        neg_files = os.listdir(neg_path) 

        # store file labels 
        map = {}
        for file in pos_files:
            map[file] = 1
        for file in neg_files:
            map[file] = 0

        '''' Downsampling to same size if needed, for class balancing '''
        num_pos_files = len(pos_files)
        num_neg_files = len(neg_files)

        print('Orig # pos files:', num_pos_files)
        print('Orig # neg files:', num_neg_files)

        if self.logger:
            
            wandb.log({"num_pos": num_pos_files})
            wandb.log({"num_neg": num_neg_files})
       
        # Downsampling 
        if num_pos_files > num_neg_files * 2:
            print('Downsampling pos files')
            pos_files = random.sample(pos_files, num_neg_files)
        elif num_neg_files  > num_pos_files * 2:
            print('Downsampling neg files')
            neg_files = random.sample(neg_files, num_pos_files)

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
    def __init__(self, cfg):
        self.dataset = ApneaDataset(cfg)
        self.batch_size = int(cfg.batch_size)
        self.train_data, self.val_data = self.dataset.get_splits(cfg.test_frac)

    def get_data(self):
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       drop_last=True)
        self.val_loader = DataLoader(self.val_data,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     drop_last=True)
        return self.train_loader, self.val_loader




''' Helper function to create directory '''
def init_dir(path): 
    if os.path.isdir(path): shutil.rmtree(path)
    if not os.path.isdir(path):
        os.makedirs(path)