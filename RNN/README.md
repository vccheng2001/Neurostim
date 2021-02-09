LSTM-RNN model to make sleep apnea predictions
Standard file format: 
- Positive/negative sequences used for testing. 
- 8 hz (8 samples/second), 10 seconds before apnea, 5 seconds after

Types of apnea:
- osa, osahs (hypopnea)...

1. Preprocessing
 
   preprocessing.py: Takes positive/negative sequences in <raw> directory and 
   cleans/processes it, outputs them to train dir

        params: <apnea_type>, <timesteps> 
        Example: python3 preprocessing.py osa 160

2. Training 

   a) rnn_train_only.py: Only trains, saves trained model to the file
      "trained_<apnea-type>_model".

        params: <apnea_type>, <timesteps> 
        Example: python3 rnn_train_only.py osa 160

   b) rnn_train_and_test.py (see below)

3. Testing
   a) rnn_train_and_test.py 
      Reads the preprocessed pos/negfiles in the train directory, trains the model, then tests it against the pos/neg files in the test directory. 

        params: <apnea_type>, <timesteps>, <threshold>
        Example: python3 rnn_train_and_test.py osa 160 0.9

        Note: in main() choose either:
        -make_predictions: makes and saves predictions
        -run_experiments: trains/tests <repeat> times, outputs
                        accuracy 
    
    b) rnn_test_only.py: Makes predictions on pos/neg files in the 
        test directory using the pre-trained model. 

        params: <apnea_type>, <timesteps>, <threshold>
        Example: python3 rnn_train_and_test.py osa 160 0.9

    c) rnn_test_window.py: takes an input test file and 
    and runs it against the trained model (trained_<apnea-type>_model).
    Performs a sliding window (window size: <timesteps>) over the test file
    and outputs a prediction file.

        params: <apnea_type>, <timesteps>, <threshold>
        Example: python3 rnn_train_and_test.py osa 160 0.9


  4. Graphing 
  For sliding window, graphs out predicted vs actual  