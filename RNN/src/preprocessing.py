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

def main():
    ''' Preprocess raw apnea files '''
    initialize_directories()
    for label in labels:
        setup_train_data(raw_path, label)
    for label in labels:
        num_files = len(os.listdir(raw_path + label))
        print(f"Parsed {str(num_files)} {label[:-1]} sequences.")

def initialize_directories():
    '''Sets up directories for train, test data '''
    init_dir(train_path)
    init_dir(test_path)
    for label in labels:
        init_dir(train_path+label)
        init_dir(test_path+label) # Comment out if sliding window

def setup_train_data(raw_path,label):
    '''Preprocesses raw data into train data'''
    dirs = raw_path + label
    files = os.listdir(dirs) 
    num_train = len(files)* 0.8 # use 80% for train

    # Read each file 
    i = 0
    for file_name in files:
        file_path = f"{dirs}/{file_name}"
        # train test split 
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
            i+=1
        except Exception as e:
            print(f"Error: {e}")
            os.remove(file_path)
            break

# Initializes directory
def init_dir(path): 
    shutil.rmtree(path)
    if not os.path.isdir(path):
        print("Making directory.... " + path)
        os.mkdir(path)
    

if __name__ == "__main__":
    main()