# pytorch lstm for apnea detection
import numpy as np
import pandas as pd
import os
import sys
import random
import torch 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import wandb


''' 
This file defines a custom Pytorch Dataset and Dataloader to be used for processing
samples to load into the model for training/testing.


ApneaDataset: 
------------------------------------------------------------------------------------
* Builds a Dataset abstraction from positive/negative sequences from the onset 
extraction algorithm
* Defines function for retrieving one sample from the dataset
* Split into training/testing dataset 


ApneaDataloader: 
-------------------------------------------------------------------------------------
* Batching samples for model input
* Customizing order of sample selection 
* Supports multiprocessing sample loading



For more information on Pytorch Datasets/Dataloaders see below:
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
 '''



class ApneaDataset(Dataset):
    ''' load the dataset config (Config is defined in apnea_detection.py) '''
    def __init__(self, cfg):
        self.pos_dir = cfg.positive_dir
        self.neg_dir = cfg.negative_dir
        self.root_dir = cfg.root_dir
        self.logger = cfg.logger
        self.data_root = os.path.join(self.root_dir, "data/")
        self.dataset = cfg.dataset
        self.apnea_type = cfg.apnea_type
        self.excerpt = cfg.excerpt
        self.test_frac = float(cfg.test_frac)
        self.timesteps = int(cfg.sample_rate) * (int(cfg.seconds_before_apnea) + int(cfg.seconds_after_apnea))
        self.path = f"{self.data_root}{cfg.dataset}/postprocessing/excerpt{cfg.excerpt}"
        self.data, self.label, self.files = self.build_data(self.path)

    ''' return size of dataset '''
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

   
    ''' build positive, negative files '''
    def build_data(self, path):
        print(f'Extracting pos/neg sequences from: {self.path}')

    
        if self.pos_dir: # can hardcode a directory to pull positive files from 
            pos_path = self.pos_dir
        else:
            pos_path = os.path.join(path, "positive")

        if self.neg_dir: # can hardcode a directory to pull negative files from 
            neg_path = self.neg_dir
        else:
            neg_path = os.path.join(path, "negative")


        # data: stores sequences
        # label: stores positive/negative labels (1/0)
        # files: store file names
        data, label, files = [], [], []


        pos_files = os.listdir(pos_path)
        neg_files = os.listdir(neg_path) 

        # map files to their labels 
        map = {}
        for file in pos_files:
            map[file] = 1
        for file in neg_files:
            map[file] = 0

        '''' Downsampling to same size if needed, for class balancing '''
        num_pos_files = len(pos_files)
        num_neg_files = len(neg_files)

        print('Number of positive files before downsampling:', num_pos_files)
        print('Number of negative files before downsampling', num_neg_files)

        if self.logger:
            
            wandb.log({"num_pos": num_pos_files})
            wandb.log({"num_neg": num_neg_files})
       
        # Downsampling if too skewed
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

            # load files along with their labels, filenames into dataset.
            # check to make sure number of timesteps (rows in file) is correct

            arr = np.loadtxt(f,delimiter="\n", dtype=np.float64)

            if arr.shape[0] >=  self.timesteps:
                arr = arr[:self.timesteps]
                data.append(np.expand_dims(arr,-1))
                label.append(map[file])
                files.append(file)
            else:
                print(f'Error: not enough timesteps in file {file}')
                exit(0)

        print(f'Number of positive files after downsampling: {len(pos_files)}')
        print(f'Number of negative files after downsampling: {len(neg_files)}')

        return data, label, files


''' Dataloader class is Pytorch's generic utility \
    in charge of batching samples from dataset, customizing sample selection order'''
class ApneaDataloader(DataLoader):
    ''' defines train/val datasets, batch size'''
    def __init__(self, cfg):

        # dataset 
        self.dataset = ApneaDataset(cfg)

        # batch size to use when loading samples
        self.batch_size = int(cfg.batch_size)

        # splits dataset into train/validation datasets based on test_frac
        self.train_data, self.val_data = self.dataset.get_splits(cfg.test_frac)


    def get_data(self):
        # dataloader for training dataset 
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       drop_last=True)
        
        # dataloader for validation dataset 
        self.val_loader = DataLoader(self.val_data,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     drop_last=True)

        return self.train_loader, self.val_loader

