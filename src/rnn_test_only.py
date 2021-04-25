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

# Visualization 
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
# from run import data, apnea_type, timesteps, batch_size, threshold

(program, data, apnea_type, timesteps, batch_size, threshold) = sys.argv
test_path = f"../{data}/TEST/test_{apnea_type}/"
pred_path = f"../{data}/PREDICTIONS/"
model_path = f"../{data}/MODELS/"
labels = {"positive/":1, "negative/":0}

def main():
    ''' Testing on unseen positive/negative sequences '''
    # load saved model 
    model = keras.models.load_model(f"{model_path}trained_{apnea_type}_model")
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
    main()