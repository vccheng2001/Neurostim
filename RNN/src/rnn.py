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
import argparse

def main():
    preprocess()
    # Train 
    model = train_model()
    # Test
    test_model(model)

################################################################################
#                               PREPROCESSING 
################################################################################
def preprocess():
    initialize_directories()
    for label in labels:
        setup_train_data(raw_path, label)
    for label in labels:
        num_files = len(os.listdir(raw_path + label))
        print(f"Parsed {num_files} {label[:-1]} sequences.")
  

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
            if df.shape[0] == int(timesteps):
                df.to_csv(out_file, index=False, header=None, sep="\n", float_format='%.4f')
        except Exception as e:
            print(f"Error: {e}")
            os.remove(file_path)
            break
        i+=1
# Initializes directory
def init_dir(path): 
    if os.path.isdir(path): shutil.rmtree(path)
    if not os.path.isdir(path):
        print("Making directory.... " + path)
        os.mkdir(path)
    
################################################################################
#                              TRAINING
################################################################################

import os, sys
import numpy as np
from numpy import mean, std, dstack 
from pandas import read_csv

# Keras LSTM model 
import tensorflow.keras as keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
# Optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import optimizers 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# Graphing 
from matplotlib import pyplot

def train_model():
    ''' Univariate LSTM model 
        1 hidden layer, binary output
    '''
    trainX, trainy = load_train_dataset() 
    n_outputs = trainy.shape[1]
    n_features = 1 # univariate 
    # Add one layer at a time 
    model = Sequential()
    # Inputs: A 3D tensor with shape [batch, timesteps, feature].
    model.add(LSTM(128, recurrent_dropout=0.2,input_shape=(int(timesteps),n_features),return_sequences=True))
    model.add(LSTM(64, input_shape=(timesteps,n_features)))
    # dense neural net layer, one output 
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    # fit network
    model.fit(trainX, trainy, epochs=int(epochs), batch_size=int(batch_size))
    model.save(f'{model_path}trained_{apnea_type}_model', overwrite=True)   # Save model 
    return model

def retrain_model():
    ''' Loads and retrains saved model '''
    trainX, trainy = load_train_dataset()
    model_name = f"{model_path}trained_{apnea_type}_model"
    print(f"Retraining model....{model_name}")
    model = keras.models.load_model(model_name)
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=0)
    return model 

def load_train_dataset():
    ''' loads train files for positive and negative sequences '''
    trainy = np.array([], dtype=np.int64)
    trainX = np.array([], dtype=np.float64).reshape(0,int(timesteps))

    # Load train files for positive and negative sequences 
    for label in labels: 
        path = train_path+label
        files = os.listdir(path)
        for file in files:
            # Append sample x to X matrix 
            sample = np.loadtxt(path + file,delimiter="\n", dtype=np.float64)
            trainX = np.vstack((trainX, sample))
            # Append binary label to y vector 
            trainy = np.hstack((trainy, labels[label]))
    trainX = np.expand_dims(trainX, axis=2)
    # convert y into a two-column probability distribution (-, +)
    trainy = to_categorical(trainy)
    return trainX, trainy


################################################################################
#                                 TESTING
################################################################################

def test_model(model):
    ''' Testing on unseen positive/negative sequences '''
    # load test data
    testX, actual = load_test_dataset()
    # get predicted class
    probabilities = model.predict(testX)
    ones = probabilities[0:,1]
    # label as 1 if predicted probability of apnea event > threshold, else label as 0
    predicted = np.where(ones > float(threshold), 1, 0)

    # make dimensions match 
    actual      = np.expand_dims(actual, axis=1)
    predicted   = np.expand_dims(predicted, axis=1)
    # evaluate accuracy, confusion matrix 
    report = summarize_results(probabilities, actual, predicted)
    return report

def predict_threshold(prob):
    '''Return 1 if probability that apnea event occurs >= threshold, else 0'''
    return 1 if prob[1] >= threshold else 0

def summarize_results(probabilities, actual, predicted):
    ''' Save predictions, confusion matrix to file '''
    report = classification_report(actual, predicted, labels=[1,0])
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0,1]).ravel()
 
    with open(f'{pred_path}predictions_{apnea_type}.txt', "w") as out:
        out.write(f"Dataset: {data}, Excerpt: {apnea_type}\n")
        out.write(f"********************************************************\n")
        out.write(f"Results: \n {report} \n")
        out.write(f" TP: {tp}\n TN: {tn}\n FP: {fp}\n FN: {fn}\n")
        out.write(f"********************************************************\n")

    with open(f'{pred_path}predictions_{apnea_type}.txt', "a") as out:
        # save to output file 
        predictions_and_labels = np.hstack((probabilities, predicted, actual))
        np.savetxt(out, predictions_and_labels, delimiter=' ',fmt='%10f', \
            header="Negative | Positive | Predicted | Actual")
    out.close()
    return report

def load_test_dataset():
    ''' Load input test vector '''
    actual = np.array([], dtype=np.int64)
    testX = np.array([], dtype=np.float64).reshape(0, int(timesteps))
    
    # Load test files 
    for label in labels:
        path = test_path+label
        files = os.listdir(path)
        for file in files:
            # print('Processing test file:', file)
            sample  = np.loadtxt(path+file,delimiter="\n", dtype=np.float64)
            testX   = np.vstack((testX, sample))
            # Append ground truth label to actual vector 
            actual  = np.hstack((actual, labels[label]))  

    testX = np.expand_dims(testX, axis=2) 
    return testX, actual 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data",  help="specify dataset (dreams, dublin, or mit)")
    parser.add_argument("-a", "--apnea_type",help="specify excerpt to use")
    parser.add_argument("-t", "--timesteps", help="specify length of sequence")
    parser.add_argument("-ep", "--epochs", help="specify number of epochs to train")
    parser.add_argument("-b", "--batch_size", help="specify batch size")
    parser.add_argument("-th", "--threshold", help="specify fraction between 0 and 1. if the predicted probability \
                                     is greater than this threshold then an apnea event is predicted.")
    args = parser.parse_args()

    # store args 

    data = args.data,
    apnea_type = args.apnea_type
    timesteps = args.timesteps
    epochs = args.epochs
    batch_size =  args.batch_size
    threshold = args.threshold
    labels ={'positive/':1, 'negative/':0}


    raw_path =      f"../{args.data}/RAW/raw_{args.apnea_type}/"
    train_path =    f"../{args.data}/TRAIN/train_{args.apnea_type}/"
    test_path =     f"../{args.data}/TEST/test_{args.apnea_type}/"
    model_path =    f"../{args.data}/MODELS/trained_{args.apnea_type}_model"
    pred_path =     f"../{args.data}/PREDICTIONS/predictions_{args.apnea_type}.txt"

    file_map = {}
    main()