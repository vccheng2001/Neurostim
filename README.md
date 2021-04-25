# LSTM-RNN model to make sleep apnea predictions from patient breathing data

## Standard file format: 
- Positive/negative sequences used for testing. 
- Datasets: MIT BIH, DREAMS, UCDDB 
- 8 hz (8 samples/second), 10 seconds before apnea, 5 seconds after onset
- types of apnea: obstructive sleep apnea (osa), hypopnea (osahs)


## End to End Apnea Prediction 
 

Args: data, apnea_type, excerpt, timesteps, epochs, batch_size, prediction_threshold

Run python3 apnea.py -h/--help for detailed arguments information 




## Running files individually (old version)

1. Preprocessing: preprocessing.py
 
     Preprocesses raw files into training data/test data using train/test split. 

     args: <data> <apnea_type>, <timesteps> 
     Example: python3 dreams preprocessing.py osa 160

2. Training: rnn_train_only.py

      Loads positive/negative sequences from preprocessed training files
      then trains/saves an RNN model to the file trained_<apnea-type>_model.

      args: <data> <apnea_type>, <timesteps> <epochs> <batch_size  
      Example: python3 rnn_train_only.py dreams osa 160 10 16

3. Testing: rnn_test_only.py
  

      Makes predictions on unseen sequences using the pre-trained model. 

      args: <data> <apnea_type>, <timesteps>, <batch_size>, <threshold>
      Example: python3 rnn_test_only.py dreams osa 160 0.7
