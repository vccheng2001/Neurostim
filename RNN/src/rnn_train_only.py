'''
rnn_train_only.py

This program reads the preprocessed train files into X, y matrices,
then trains/saves an RNN model to the file trained_<apnea-type>_model.

params: <apnea_type>, <timesteps> <epochs> <batch_size  
Example: python3 rnn_train_only.py osa 160 10 16
'''

import os, sys
import numpy as np
from numpy import mean, std, dstack 
from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras import optimizers 

# Keras LSTM model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as keras 
# Graphing 
from matplotlib import pyplot

# parameters
(program, data, apnea_type, timesteps, epochs, batch_size) = sys.argv
timesteps, epochs, batch_size = int(timesteps), int(epochs), int(batch_size)
labels = {"positive/":1, "negative/":0}
train_group = f"../{data}/TRAIN/train_{apnea_type}/"
pred_path = f"../{data}/PREDICTIONS/"
model_path = f"../{data}/MODELS/"

# fit and evaluate rnn-lstm model
def build_model(trainX, trainy):
    # trainX, trainy = load_train_dataset()       # Load train data 
    n_samples, n_timesteps, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # Add one layer at a time 
    model = Sequential()
    # 100 units in output 
    model.add(LSTM(128, input_shape=(n_samples,n_timesteps)))
    # drop 50% of input units 
    model.add(Dropout(0.1))
    # dense neural net layer, relu(z) = max(0,z) output = activation(dot(input, kernel)
    model.add(Dense(32, activation='relu'))
    # Softmax: n_outputs in output (1)
    model.add(Dense(n_outputs, activation='sigmoid'))
    # Binary 0-1 loss, use SGD 
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size)
    return model

# load train files for positive and negative sequences 
def load_train_dataset():
    # Load Train Data 
    trainy = np.array([],dtype=np.int64)
    trainX = np.array([], dtype=np.float64).reshape(0,int(timesteps))
    # Load train files for positive and negative sequences 
    for label in labels:
        trainX, trainy = load_files_train(label, trainX, trainy)
    trainX = np.expand_dims(trainX, axis=2)
    # convert y into a two-column probability distribution (-, +)
    trainy = to_categorical(trainy)
    return trainX, trainy

# Creates X, y train matrices 
def load_files_train(label, trainX, trainy):
    path = train_group+label # e.g. train/positive/
    files = os.listdir(path)
    # Load all N train files one sample at a time
    for file in files:
        sample= np.loadtxt(path + file,delimiter="\n", dtype=np.float64)
        # print(trainX.shape, sample.shape)
        trainX = np.vstack((trainX, sample))
        # Append binary 1 or 0 to y vector 
        binary_label = labels[label]
        trainy = np.hstack((trainy,binary_label))
    return trainX, trainy

def main():
    trainX, trainy = load_train_dataset()       # Load train data 
    model = build_model(trainX, trainy)
    # # Comment this out for retraining
    # model = KerasClassifier(build_fn=build_model, verbose=0)
    # batch_size = [8, 16, 32, 64]
    # epochs = [10, 15, 20]
  
  
    # param_grid = dict(batch_size=batch_size, epochs=epochs)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    # grid_result = grid.fit(trainX, trainy)


    # # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))


    # Comment this out unless retrain 
    # model = retrain_model(trainX, trainy)
# 
    model.save(f'{model_path}trained_{apnea_type}_model', overwrite=True)   # Save model 

# Retrain 
def retrain_model(trainX, trainy):
    model_name = f"{model_path}trained_{apnea_type}_model"
    print(f"Retraining model....{model_name}")
    model = keras.models.load_model(model_name)
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size)
    return model 

if __name__ == "__main__":
    main()