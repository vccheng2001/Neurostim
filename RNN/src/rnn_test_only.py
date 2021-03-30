'''
rnn_test_only.py

This program makes predictions on files in the 
test directory using the pre-trained model. 
** must train first 

params: <data> <apnea_type>, <timesteps>, <batch_size>, <threshold>
Example: python3 rnn_test_only.py dreams osa 160 0.7
'''
import glob
import os, sys
import numpy as np
from numpy import mean, std, dstack 
from pandas import read_csv

# Keras LSTM model 
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

# Graphing 
from matplotlib import pyplot

from sklearn.metrics import confusion_matrix, classification_report
# parameters 

(program, data, apnea_type, timesteps, batch_size, threshold) = sys.argv
test_path = f"../{data}/TEST/test_{apnea_type}/"
pred_path = f"../{data}/PREDICTIONS/"
model_path = f"../{data}/MODELS/"

labels = {"positive/":1, "negative/":0}

# tests on positive/negative sequences in test files
def main():
    # load saved model 
    model = keras.models.load_model(f"{model_path}trained_{apnea_type}_model")
    # load input test vector 
    testX, actual = load_test_dataset()

    # Accuracy
    testy = to_categorical(actual)
    print_accuracy(testX, testy, model, int(batch_size))

    predictions = model.predict(testX, int(batch_size))
    # save predictions to file 
    output_predictions(predictions, np.asarray(actual))
   
# Summarizes accuracy by comparing predicted with actual labels 
def print_accuracy(testX, testy, model, batch_size):
    _, accuracy = model.evaluate(testX, testy, batch_size)
    score = accuracy * 100.0
    print('>%.3f' % (score)) 

# saves predictions 
def output_predictions(predictions,actual):
    # number of test files in test dir
    num_predictions = predictions.shape[0] 
    # 1 if > threshold, 0 otherwise 
    flags = np.zeros((num_predictions, 1))
    for i in range(num_predictions):
        p = predictions[i]
        # flag apnea (1) if positive prediction >= threshold, else 0
        flags[i] = 1 if p[1] >= float(threshold) else 0
    
    # Add column for flagged (predicted) values
    predictions = np.hstack((predictions, flags))

    # Add column for actual values
    actual = np.expand_dims(actual, axis=1)

    print("Confusion Matrix")
    # N = actual.shape[0]
    # print(f"True Positives: { np.sum((flags==1) and (actual==1))/N}")
    # print(f"True Negatives: { np.sum((flags==0) and (actual==0))/N}")
    # print(f"False Positives: {np.sum((flags==1) and (actual==0))/N}")
    # print(f"False Negatives: {np.sum((flags==0) and (actual==1))/N}")
    predictions = np.hstack((predictions, actual))
    report = classification_report(actual,flags, labels=[1,0])
    tn, fp, fn, tp = confusion_matrix(actual,flags, labels=[0,1]).ravel()
    print(report)
    print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')
    print(predictions)
    # save predictions

    f= open(f'{pred_path}predictions_{apnea_type}.txt','w')
    f.write(report + '\n')
    f.write(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n')
    f.close()
    with open(f'{pred_path}predictions_{apnea_type}.txt', "ab") as f:
        np.savetxt(f, predictions, delimiter=' ',fmt='%10f',header="Negative,Positive,Prediction,Actual")

'''
The precision is the ratio tp / (tp + fp) where tp is the number of true positives
and fp the number of false positives. The precision is intuitively the ability of 
the classifier not to label as positive a sample that is negative.

The recall is the ratio tp / (tp + fn) where tp is the number of true positives
 and fn the number of false negatives. The recall is intuitively the ability of 
the classifier to find all the positive samples.

Recall is a metric that quantifies the number of correct positive predictions made 
out of all positive predictions that could have been made. Unlike precision that 
only comments on the correct positive predictions out of all positive predictions,
 recall provides an indication of missed positive predictions

The F-beta score can be interpreted as a weighted harmonic mean of the precision and 
recall, where an F-beta score reaches its best value at 1 and worst score at 0.

The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 
means recall and precision are equally important.

'''
# Create test X matrix, actual values
def load_files_test(actual, label, X):

    #Use this if subdirs in positive/negative dirs 
    
    # path=test_path+label + "*"
    # dirs = glob.glob(path)
    # for d in dirs:
    #     files = os.listdir(d)
    #     for file_name in files:
    #         file_path = f"{d}/{file_name}"

    #         arr = np.loadtxt(file_path,delimiter="\n", dtype=np.float64)
    #         if X.shape[1] == arr.shape[0]: # make sure dims match
    #             print(f'Currently processing test file {file_name}, {labels[label]}')
    #             # Add as row to x matrix
    #             X = np.vstack((X, arr))
    #             print(X.shape)
    #             # Build actual values (positive/negative)
    #             actual.append(labels[label])
    # return X

    # Use this if files directly in positive/negative dirs
    path=test_path+label
    files = os.listdir(path)
    for file_name in files:
        file_path = path+file_name
        # print('Currently processing test file:', file_name)
        sample = np.loadtxt(file_path,delimiter="\n", dtype=np.float64)
        # Add as row to x matrix
        X = np.vstack((X, sample))
        # Build actual values (positive/negative)
        actual.append(labels[label])
    return X


# load input test vector as matrix 
def load_test_dataset():
    actual = list()
    testX = np.array([], dtype=np.float64).reshape(0,int(timesteps))
    # Load test files 
    for label in labels:
        testX = load_files_test(actual, label, testX)
    testX = np.expand_dims(testX, axis=2)
    return testX, actual


if __name__ == "__main__":
    main()