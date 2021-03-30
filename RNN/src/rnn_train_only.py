'''
rnn_train_only.py

This program reads the positive/negative sequences from preprocessed training files
then trains/saves an RNN model to the file trained_<apnea-type>_model.

params: <data> <apnea_type>, <timesteps> <epochs> <batch_size  
Example: python3 rnn_train_only.py dreams osa 160 10 16
'''

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

# Graphing 
from matplotlib import pyplot


def main():
    trainX, trainy = load_train_dataset() 
    model = train_model(trainX, trainy)
    # model = retrain_model()
    model.save(f'{model_path}trained_{apnea_type}_model', overwrite=True)   # Save model 

def train_model(trainX, trainy):
    ''' Univariate LSTM model 
        1 hidden layer, binary output
    '''
    n_outputs = trainy.shape[1]
    n_features = 1 # univariate 
    # Add one layer at a time 
    model = Sequential()
    # Inputs: A 3D tensor with shape [batch, timesteps, feature].
    model.add(LSTM(256, input_shape=(timesteps,n_features)))
    # drop 50% of input units 
    model.add(Dropout(0.5))
    # dense neural net layer, one output 
    model.add(Dense(2, activation='sigmoid'))
    # Binary 0-1 loss, use SGD 
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size)
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


if __name__ == "__main__":
    (program, data, apnea_type, timesteps, epochs, batch_size) = sys.argv
    timesteps, epochs, batch_size = int(timesteps), int(epochs), int(batch_size)
    labels = {"positive/":1, "negative/":0}
    train_path =   f"../{data}/TRAIN/train_{apnea_type}/"
    pred_path =     f"../{data}/PREDICTIONS/"
    model_path =    f"../{data}/MODELS/"
    main()