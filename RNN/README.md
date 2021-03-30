LSTM-RNN model to make sleep apnea predictions

Standard file format: 
- Positive/negative sequences used for testing. 
- 8 hz (8 samples/second), 10 seconds before apnea, 5 seconds after

Types of apnea:
- osa, osahs (hypopnea)...

1. Preprocessing: preprocessing.py
 

     Preprocesses raw files into training data. 
     sorted by positive/negative sequences. 

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
