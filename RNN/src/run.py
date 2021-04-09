import sys
import os
import subprocess
import argparse
import datetime 
import csv

results = {}

def main():
    # ''' Main file to run preprocessing, train, and test all at once '''

    # # parse command line arguments 
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--data",  help="specify dataset (dreams, dublin, or mit)")
    # parser.add_argument("-a", "--apnea_type",help="specify excerpt to use")
    # parser.add_argument("-t", "--timesteps", help="specify length of sequence")
    # parser.add_argument("-ep", "--epochs", help="specify number of epochs to train")
    # parser.add_argument("-b", "--batch_size", help="specify batch size")
    # parser.add_argument("-th", "--threshold", help="specify fraction between 0 and 1. if the predicted probability \
    #                                  is greater than this threshold then an apnea event is predicted.")
    # args  = parser.parse_args()

    # # store args 
    # data        = args.data
    # apnea_type  = args.apnea_type
    # timesteps   = args.timesteps
    # epochs      = args.epochs
    # batch_size  = args.batch_size
    # threshold   = args.threshold

    # params = [epochs,batch_size,threshold]


    for i in (4,10):
        subprocess.call(cmd)

        # excerpt = f"{apnea_type}{i}"
        # # params: <data> <apnea_type>, <timesteps>, <batch_size>, <threshold>
        # # (program, data, timesteps, epochs, batch_size, threshold) = sys.argv
        # print(f"********************** PROCESSING {excerpt} *************************")
        # print(f"\nPreprocessing {apnea_type} \n")
        # subprocess.call(['python3', 'rnn.py', data, excerpt, timesteps])
        
        # print("\nTraining....\n")
        # subprocess.call(['python3', 'rnn_train_only.py' ,data, excerpt, timesteps, epochs, batch_size])
        
        # print("\nTesting....\n")
        # subprocess.call(['python3', 'rnn_test_only.py', data, excerpt, timesteps, batch_size, threshold],stderr=subprocess.STDOUT)
    
if __name__ == "__main__":
    main()