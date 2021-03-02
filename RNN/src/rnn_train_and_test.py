'''
rnn_train_and_test.py

This program reads the preprocessed files in the train directory,
trains the model, then tests it against the files in the test directory. 

params: <apnea_type>, <timesteps>, <threshold
Example: python3 rnn_train_and_test.py osa 160 0.9

Note: in main() choose either:
 -make_predictions: makes and saves predictions
 -run_experiments: trains/tests <repeat> times, outputs
                   accuracy 
'''

import os, sys
import numpy as np
from numpy import mean, std, dstack 
from pandas import read_csv

# Keras LSTM model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

# Graphing 
from matplotlib import pyplot

# parameters 
(program, apnea_type, timesteps, threshold) = sys.argv
test_path = f"test_{apnea_type}/"
train_path = f"train_{apnea_type}/"
labels = {"positive/":1, "negative/":0}
epochs, batch_size = 15, 64

# Load files for a given group, label (e.g. test/positive)
def load_files(label, group, X, y):
    path = group+label
    files = os.listdir(path)

    for file in files:
        # Load each x sample 
        print(f"Processing {group} file: {file}")
        sample = np.loadtxt(path + file,delimiter="\n", dtype=np.float64)
        X = np.vstack((X,sample))

        # Append binary 1 or 0 to y vector 
        binary_label = labels[label]
        y = np.hstack((y,binary_label))
    return X, y

# create X, y matrices 
def load(path):
    y = np.array([],dtype=np.int64)
    X = np.array([], dtype=np.float64).reshape(0,int(timesteps))
    for label in labels:
        X, y = load_files(label, path, X, y)
    X = np.expand_dims(X, axis=2)
    y = to_categorical(y)
    return X, y

# Load train, test datasets into matrices 
def load_train_test_datasets():
    trainX, trainy = load(train_path)
    testX, testy = load(test_path) 
    return trainX, trainy, testX, testy 

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # Add one layer at a time 
    model = Sequential()
    # 100 units in output 
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    # drop 50% of input units 
    model.add(Dropout(0.5))
    # dense neural net layer, relu(z) = max(0,z) output = activation(dot(input, kernel)
    model.add(Dense(100, activation='relu'))
    # Softmax: n_outputs in output (1)
    model.add(Dense(n_outputs, activation='softmax'))
    # Binary 0-1 loss, use SGD 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size)
    # evaluate model
    # _, accuracy = model.evaluate(testX, testy, batch_size=batch_size)
    # return accuracy
    return (model, batch_size)

####################################################################
#       makes predictions, save to output file 
####################################################################

# make predictions 
def make_predictions():
    # load train, test data
    trainX, trainy, testX, testy = load_train_test_datasets()
    # train
    model, batch_size = evaluate_model(trainX, trainy, testX, testy)
    # predict on test X 
    predictions = model.predict(testX, batch_size)
    # number of samples generated using sliding window 
    num_predictions = predictions.shape[0]
    # 1 if predict apnea, 0 otherwise 
    flags = np.zeros((num_predictions, 1))

    for i in range(num_predictions):
        p = predictions[i]
        # flag apnea (1) if positive prediction >= threshold, else 0
        flags[i] = 1 if p[1] >= float(threshold) else 0
    predictions = np.hstack((predictions, flags))
    
    # Actual y outputs 
    actual = testy[:,1]
    actual = np.expand_dims(actual,axis=1)
    predictions = np.hstack((predictions,actual))

    # Save to predictions file 
    np.savetxt(f'predictions_{apnea_type}.txt', predictions, delimiter=' ',fmt='%10f',header="Negative,Positive,Prediction,Actual")

####################################################################
#       train/test <repeat> times, outputs accuracy
####################################################################

# Run full train/test <repeats> times
def run_experiment(repeats=3):
    # load data
    trainX, trainy, testX, testy = load_train_test_datasets()
    # keep track of accuracy 
    scores = list()
    for r in range(repeats):
        model, batch_size = evaluate_model(trainX, trainy, testX, testy)
        _, accuracy = model.evaluate(testX, testy, batch_size)
        score = accuracy * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


if __name__ == "__main__":
    # makes and saves predictions 
    # make_predictions()
    # runs <repeat> times and predicts accuracies
    run_experiment()