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
import csv
import sys
import shutil

(program, data, apnea_type, timesteps) = sys.argv
raw_path =      f"../{data}/RAW/raw_{apnea_type}/"
train_path =    f"../{data}/TRAIN/train_{apnea_type}/"
test_path =     f"../{data}/TEST/test_{apnea_type}/"
labels =        ["positive/", "negative/"]

# Preprocesses apnea files 
def main():
    init_dirs()
    for label in labels:
        setup_train_data(raw_path, label)
    for label in labels:
        num_files = len(os.listdir(raw_path + label))
        print(f"Parsed {str(num_files)} {label[:-1]} sequences.")

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
    dirs = raw_path + label
    files = os.listdir(dirs) 
    num_files = len(files)
    num_train = num_files * 0.8 # use 80% for train

    # Read each file 
    i = 0
    for file_name in files:
        file_path = f"{dirs}/{file_name}"
        # print(f"Input:{file_name}") # input
        path = train_path if i < num_train else test_path
        out_file = f"{path}{label}{label[:-1]}_{str(i)}.txt"
        
        try:
            # Read raw file
            df = pd.read_csv(file_path, skip_blank_lines=True, header=None, sep="\n")
            df.dropna(axis=0,inplace=True)
            # Keep only <timesteps> rows
            df = df.head(int(timesteps))
            # print("Output:" , out_file) # output
            df.to_csv(out_file, index=False, header=None, sep="\n", float_format='%.4f')
        except Exception as e:
            print(f"Error: {e}")
            os.remove(file_path)
            continue
        i+=1


# Makes directory 
def make_dir(path):
    if not os.path.isdir(path):
        print("Making directory.... " + path)
        os.mkdir(path)


# Clears directory
def remove_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)

    

if __name__ == "__main__":
    main()