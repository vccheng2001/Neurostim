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
    # model.save(f'{model_path}{apnea_type}_{excerpt}.ckpt', overwrite=True)   # Save model 
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
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

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

    # calculate scores
    # auc = roc_auc_score(testy, ones)
    # calculate roc curves
    # fpr, tpr, _ = roc_curve(testy, ones)
    # plot the roc curve for the model
    # pyplot.plot(fpr, tpr, linestyle='--', label=ROC)
    # # axis labels
    # pyplot.xlabel('False Positive Rate')
    # pyplot.ylabel('True Positive Rate')
    # # show the legend
    # pyplot.legend()
    # show the plot    
    
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
    pred_time   = np.asarray(pred_time)
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
            
            # filename encodes the # seconds (float) since start of patient recording
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
    ''' Summarizes scores, save to file '''

    # precision, recall, fscore, support 
    p, r, f, s = precision_recall_fscore_support(actual, predicted, labels=[0,1])
    # true positives, true negatives, false positives, false negatives 
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0,1]).ravel()
    # append scores as row to csv log 
    with open(f"{info_path}summary_results_v1.csv", 'a', newline='\n') as csvfile:
        fieldnames = [  'dataset',      'apnea_type',  'excerpt', 'num_pos_train',    'num_neg_train',\
                        'precision_1',  'precision_0',  'recall_1', 'recall_0',  'f1_1', 'f1_0',\
                        'support_1','support_0','true_pos','true_neg','false_pos','false_neg',
                        'num_epochs', 'batch_size']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'dataset'      :data,
                         'apnea_type'   :apnea_type,
                         'excerpt'      :excerpt,
                         'num_pos_train':get_num_files(train_path+'positive/'),
                         'num_neg_train':get_num_files(train_path+'negative/'),
                         'precision_1'  :round(p[1],3), 'precision_0'  :round(p[0],3),
                         'recall_1'     :round(r[1],3), 'recall_0'     :round(r[0],3),
                         'f1_1'         :round(f[1],3), 'f1_0'         :round(f[0],3),
                         'support_1'    :round(s[1],3), 'support_0'    :round(s[0],3),
                         'true_pos'     :tp,            'true_neg'     :tn,
                         'false_pos'    :fp,            'false_neg'    :fn,
                         'num_epochs'   :epochs,        'batch_size'   :batch_size})
    
    predictions_and_labels = np.hstack((probabilities, predicted, actual, pred_time))
    # save prediction to output file 
    with open(f'{pred_path}{apnea_type}_{excerpt}_v1.csv', "w") as out:
        np.savetxt(out, predictions_and_labels, delimiter=',',fmt='%s', \
            header="prob_neg,prob_pos,predicted,actual,timestamp", comments='')
    out.close()


def get_num_files(path):
    ''' return number of files in a directory '''

    return len(os.listdir(path))

def parse_patient_start_times(apnea_type, excerpt):
    ''' parses start time of spectrogram recording corresponding to current patient '''

    df = pd.read_csv(f"{info_path}patient_start_times.csv",header=0,index_col=False)
    # get start_time value 
    df = df[(df['data'] == str(data)) & (df['excerpt'] == int(excerpt))]
    start_time = df["start_time"].item()
    # convert string to datetime, format is hh:mm:ss
    start_time = datetime.strptime(start_time, '%H:%M:%S')
    return start_time



if __name__ == "__main__":
    ''' parses command line arguments, runs main() '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data",         help="dataset (dreams, dublin, or mit)")
    parser.add_argument("-a", "--apnea_type",   help="type of apnea (osa, osahs, or all)")
    parser.add_argument("-ex","--excerpt",      help="excerpt number to use")
    parser.add_argument("-t", "--timesteps",    help="length of sequence in timesteps")
    parser.add_argument("-ep","--epochs",       help="number of epochs to train")
    parser.add_argument("-b", "--batch_size",   help="batch size")
    parser.add_argument("-th","--threshold",    help="threshold fraction for predicting positive apnea")
    # parse args 
    args = parser.parse_args()

    print(args)
    # store args 
    data        = args.data
    apnea_type  = args.apnea_type
    excerpt     = args.excerpt
    timesteps   = args.timesteps
    epochs      = args.epochs
    batch_size  = args.batch_size
    threshold   = args.threshold
    labels      = {'positive/':1, 'negative/':0}
    train_frac  = 0.7 # default ratio for train-test-split

    print(f"Processing {data}: {apnea_type}_{excerpt}_v1")

    raw_path =      f"../{data}/RAW/{apnea_type}_{excerpt}/"
    train_path =    f"../{data}/TRAIN/{apnea_type}_{excerpt}/"
    test_path =     f"../{data}/TEST/{apnea_type}_{excerpt}/"
    model_path =    f"../{data}/MODELS/"
    pred_path =     f"../{data}/PREDICTIONS/"
    info_path =     f"../info/"
    # timestamp of recording start time
    start_time =    parse_patient_start_times(apnea_type, excerpt)
    main()
