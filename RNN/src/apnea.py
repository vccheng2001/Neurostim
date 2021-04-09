''' 
This program makes sleep apnea predictions on three medical datasets: DREAMS,
University of Dublin (ucddb), and MIT-BIH data. 
    1) preprocesses data 
    2) trains
    3) tests on unseen data 

    Requirements: tensorflow 2.0+, python3
    Run python3 apnea.py -h/--help for command line arguments and usage details
'''

import pandas as pd 
import os
import csv
import sys
import shutil
import argparse
import re 
from datetime import datetime, timedelta
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    preprocess()
    # Train 
    model = train_model()
    # Test
    test_model(model, start_time)

'''###############################################################################
#                               PREPROCESSING 
################################################################################'''
def preprocess():
    ''' Preprocesses and organizes raw data into train data, test data'''
    initialize_directories()
    for label in labels:
        setup_train_data(raw_path, label)
    for label in labels:
        num_files = get_num_files(raw_path + label)
        print(f"Parsed {num_files} {label[:-1]} sequences.")
  

def initialize_directories():
    '''Sets up directories for train, test data '''
    init_dir(train_path)
    init_dir(test_path)
    for label in labels:
        init_dir(train_path+label)
        init_dir(test_path+label) # Comment out if sliding window

def setup_train_data(raw_path,label):
    ''' performs data cleaning, train-test-split '''
    dirs = raw_path + label
    files = os.listdir(dirs) 
    num_train = len(files) * train_frac # use 80% for train

    # Read each file 
    i = 0
    for file_name in files:
        file_path = f"{dirs}/{file_name}"
        # train test split 
        path = train_path if i < num_train else test_path
        out_file = f"{path}{label}{label[:-1]}_{str(i)}-[{file_name[:-4]}].txt"
        try:
            # Read raw file
            df = pd.read_csv(file_path, skip_blank_lines=True, header=None, sep="\n")
            df.dropna(axis=0,inplace=True)
            # Keep only <timesteps> rows
            df = df.head(int(timesteps))
            if df.shape[0] == int(timesteps):
                df.to_csv(out_file, index=False, header=None, sep="\n", float_format='%.4f')
        except Exception as e:
            print(f"Error: {e}")
            os.remove(file_path)
            break
        i+=1

def init_dir(path): 
    ''' initialize directory '''
    if os.path.isdir(path): shutil.rmtree(path)
    if not os.path.isdir(path):
        print("Making directory.... " + path)
        os.mkdir(path)
    
'''################################################################################
#                              TRAINING
################################################################################'''

import numpy as np

# Keras LSTM model 
import tensorflow.keras as keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
# Optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import optimizers 

def train_model():
    ''' fits data to univariate LSTM model with 1 hidden layer, binary output
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
    model.save(f'{model_path}trained_{apnea_type}_model_NEW', overwrite=True)   # Save model 
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


'''################################################################################
#                                 TESTING
################################################################################'''
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
# Graphing 
from matplotlib import pyplot

def test_model(model, start_time):
    ''' Testing on unseen positive/negative sequences '''
    # load test data
    testX, actual, file_map = load_test_dataset()
    # get predicted class
    probabilities = model.predict(testX)
    num_test = len(probabilities)
    ones = probabilities[0:,1]
    # label as 1 if predicted probability of apnea event > threshold, else label as 0
    predicted = np.where(ones > float(threshold), 1, 0)
    
    pred_time = []
    for i in range(num_test):
        # if wrong prediction, mark timestamp in the format hh:mm:ss
        if predicted[i] != actual[i]:
            # start time of patient recording
            start_time = str(start_time).split(' ')[1]
            start_time = datetime.strptime(str(start_time), '%H:%M:%S')
            # offset from start of patient recording in hh:mm:ss
            apnea_offset = file_map[i]
            # compute absolute timestamp of prediction in hh:mm:ss format
            prediction_time = (start_time + apnea_offset).strftime('%H:%M:%S')
            pred_time.append(prediction_time)
        
        else:  
            # no need to mark timestamp if prediction is correct  
            pred_time.append("")              
            

    # make dimensions match 
    actual      = np.expand_dims(actual, axis=1)
    predicted   = np.expand_dims(predicted, axis=1)
    pred_time = np.asarray(pred_time)
    pred_time   = np.expand_dims(pred_time, axis=1)

    # evaluate accuracy, confusion matrix 
    summarize_results(probabilities, actual, predicted, pred_time)


def load_test_dataset():
    file_map = []
    ''' Load input test vector '''
    actual = np.array([], dtype=np.int64)
    testX = np.array([], dtype=np.float64).reshape(0, int(timesteps))
    
    # Load test files 
    for label in labels:
        path = test_path+label
        files = os.listdir(path)
        for file in files:
            
            # filename encodes the # seconds since start of patient recording
            # parse seconds (format xxx.xxx) 
            apnea_offset_sec = re.search(r'\[(.*?)\]', file).group(1)
            apnea_offset_sec = apnea_offset_sec.split(".")[:2]
            apnea_offset_sec = ".".join(apnea_offset_sec)  

            # convert seconds to datetime format hh:mm:yy, store into file_map
            apnea_offset = timedelta(seconds=float(apnea_offset_sec))
            file_map.append(apnea_offset)            

            # Append test input to vector 
            sample  = np.loadtxt(path+file,delimiter="\n", dtype=np.float64)
            testX   = np.vstack((testX, sample))
            # Append ground truth label to actual vector 
            actual  = np.hstack((actual, labels[label]))  

    file_map = dict(enumerate(file_map))
    testX = np.expand_dims(testX, axis=2) 
    return testX, actual, file_map

def summarize_results(probabilities, actual, predicted, pred_time):
    ''' Save predictions, confusion matrix to file '''

    # precision, recall, fscore, support 
    p, r, f, s = precision_recall_fscore_support(actual, predicted, labels=[0,1])
    # true positives, true negatives, false positives, false negatives 
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0,1]).ravel()
    # append scores as row to csv log 
    with open(f"{info_path}results.csv", 'a', newline='\n') as csvfile:
        fieldnames = [  'dataset',      'apnea_type',   'num_pos_train',    'num_neg_train',\
                        'precision_1',  'precision_0',  'recall_1', 'recall_0',  'f1_1', 'f1_0',\
                        'support_1','support_0','true_pos','true_neg','false_pos','false_neg']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'dataset': data,
                         'apnea_type':apnea_type,
                         'num_pos_train':get_num_files(train_path+'positive/'),
                         'num_neg_train':get_num_files(train_path+'negative/'),
                         'precision_1'  :p[1], 'precision_0'  :p[0],
                         'recall_1'     :r[1], 'recall_0'     :r[0],
                         'f1_1'         :r[1], 'f1_0'         :r[0],
                         'support_1'    :s[1], 'support_0'    :s[0],
                         'true_pos'     :tp,   'true_neg'     :tn,
                         'false_pos'    :fp,   'false_neg'    :fn})
    
    predictions_and_labels = np.hstack((probabilities, predicted, actual))#, pred_time))
    # save prediction to output file 
    with open(f'{pred_path}predictions_{apnea_type}.csv', "w") as out:
        np.savetxt(out, predictions_and_labels, delimiter=',',fmt='%s', \
            header="prob_neg,prob_pos,predicted,actual,timestamp")
    out.close()


def get_num_files(path):
    ''' return number of files in a directory '''
    return len(os.listdir(path))

def parse_record_start_times(info_path, apnea_type):
    ''' parses start time of spectrogram recording corresponding to current excerpt '''
    df = pd.read_csv(f"{info_path}record_start_times.csv",header=0,index_col=False)
    # get excerpt number 
    excerpt = re.search(r'(\d+)$', apnea_type).group(1)
    # get start_time value 
    df = df[(df['data'] == str(data)) & (df['excerpt'] == int(excerpt))]
    start_time = df["start_time"].item()
    # convert string to datetime, format is hh:mm:ss
    start_time = datetime.strptime(start_time, '%H:%M:%S')
    return start_time



if __name__ == "__main__":
    ''' parses command line arguments, runs main() '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data",         help="specify dataset (dreams, dublin, or mit)")
    parser.add_argument("-a", "--apnea_type",   help="specify excerpt to use")
    parser.add_argument("-t", "--timesteps",    help="specify length of sequence")
    parser.add_argument("-ep", "--epochs",      help="specify number of epochs to train")
    parser.add_argument("-b", "--batch_size",   help="specify batch size")
    parser.add_argument("-th", "--threshold",   help="specify fraction between 0 and 1.\
    if the predicted probability is greater than this threshold then an apnea event is predicted.")

    # parse args 
    args = parser.parse_args()

    # store args 
    data = args.data
    apnea_type = args.apnea_type
    timesteps = args.timesteps
    epochs = args.epochs
    batch_size =  args.batch_size
    threshold = args.threshold
    labels ={'positive/':1, 'negative/':0}
    train_frac = 0.8 # default ratio for train-test-split

    for i in [1,2,3,4,5,6,7,8,9]:
        apnea_type = f"{args.apnea_type}_excerpt{i}"
        print(f"Processing {apnea_type}")
 
        raw_path =      f"../{data}/RAW/raw_{apnea_type}/"
        train_path =    f"../{data}/TRAIN/train_{apnea_type}/"
        test_path =     f"../{data}/TEST/test_{apnea_type}/"
        model_path =    f"../{data}/MODELS/"
        pred_path =     f"../{data}/PREDICTIONS/"
        info_path =     f"../info/"
        # timestamp of recording start time
        start_time =    parse_record_start_times(info_path, apnea_type)
        main()
