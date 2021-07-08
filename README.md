# End-to-End Sleep Apnea Prediction using Deep Learning
***
This is a end-to-end apnea detection program using a deep learning model
which performs the following steps:
Preprocessing:
   - data normalization, visualization, onset events extraction
Model Training, Testing

## File structure
```
data/:  includes all dataset files (note: raw data files must be of the form <dataset>_<apnea_type>_ex<excerpt>_sr<sample_rate>.txt
info/:  metadata
neurostim/: django webapp 
onset_extraction.py: extracts onset/non-onset events, generate +/- sequences
apnea_detection.py: main program to run pipeline 
cnn.py: defines 1D CNN model
lstm.py: defines LSTM model 
train.py: performs model training/evaluation 
dataloader.py: builds dataloader from dataset, train-test split
```

 * data/
   * dreams/
       * preprocessing/
          * excerpt1/
            * dreams_osa_ex1_sr8.csv
          * excerpt2/
       * postprocessing/
          * excerpt1/
            * positive
            * negative
          * excerpt2/
   * mit/
   * dublin/
   * patch/
 * info/
 * neurostim/
 * README.md

- Datasets supported: mit, dreams, ucddb, patch
- types of apnea: obstructive sleep apnea (osa)



## Installation
 ```bash
 git clone https://github.com/vccheng2001/Apnea-Detection-LSTM.git
 cd Apnea-Detection-LSTM/ 
 git checkout upgrade_v2
 pip3 install -r requirements.txt  (install all dependencies)
 ```

## Running without the webapp (updated)
 
 python3 apnea_detection.py 

 (Configure DefaultConfig() in the same file to change default parameters)

## Running with the webapp (not updated as of 7/8)
 ```bash

 cd neurostim
 cd apnea_detection
 python3 manage.py runserver 
 
 ```
 
 
