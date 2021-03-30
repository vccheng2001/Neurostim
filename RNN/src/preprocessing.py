'''
preprocessing.py

This program preprocesses raw files into training data,
sorted by positive/negative sequences. 

params: <data> <apnea_type>, <timesteps> 
Example: python3 dreams preprocessing.py osa 160
'''
import glob
import pandas as pd 
import os
import numpy as np
import csv
import shutil
import sys


(program, data, apnea_type, timesteps) = sys.argv
raw_path = f"../{data}/RAW/raw_{apnea_type}/"
train_path = f"../{data}/TRAIN/train_{apnea_type}/"
test_path = f"../{data}/TEST/test_{apnea_type}/"
labels = ["positive/", "negative/"]

# Preprocesses apnea files 
def main():
    init_dirs()
    for label in labels:
        setup_train_data(raw_path, label)
    for label in labels:
        num_files = len(os.listdir(raw_path + label))
        print(f"Number of {label[:-1]}s: {str(num_files)}")

# Sets up directories for train, test data 
def init_dirs():
    remove_dir(train_path)
    remove_dir(test_path)
    make_dir(train_path)
    make_dir(test_path)
    for label in labels:
        make_dir(train_path+label)
        make_dir(test_path+label) # Comment out if sliding window

# Preprocesses raw data into train data
def setup_train_data(raw_path,label):
    # rootdir = raw_path + label + "*"
    # dirs = glob.glob(rootdir)

    # for d in dirs:
    #     files = os.listdir(d)

    dirs = raw_path + label
    files = os.listdir(dirs) 
    num_files = len(files)
    num_train = num_files * 0.8

    # Read each file 
    i = 0
    for file_name in files:
        file_path = f"{dirs}/{file_name}"
        # Input raw file
        # print(f"Input:{file_name}")

        # train-test-split 
        if i < num_train: # train 80%
            out_file = f"{train_path}{label}{label[:-1]}_{str(i)}.txt"
        else:             # test 20%
            out_file = f"{test_path}{label}{label[:-1]}_{str(i)}.txt"

        try:
            # Read raw file
            df = pd.read_csv(file_path, skip_blank_lines=True, header=None, sep="\n")
            df.dropna(axis=0,inplace=True)
            df = df.head(int(timesteps))
            # Output 
            if not df.empty and df.shape[0] == int(timesteps):
                # print("Output:" , out_file)
                df.to_csv(out_file, index=False, header=None,sep="\n",float_format='%.4f')

        except Exception as err:
            print(err)
            os.remove(file_path)
            continue
        i+=1


# Clears directory
def remove_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)

# Makes directory 
def make_dir(path):
    if not os.path.isdir(path):
        print("Making dir.... " + path)
        os.mkdir(path)
    

if __name__ == "__main__":
    main()