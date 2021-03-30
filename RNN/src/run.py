import sys
import os
import subprocess
import argparse
import datetime 
import csv

results = {}

def main():
    ''' Main file to run preprocessing, train, and test all at once '''

    # parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data",  help="specify dataset (dreams, dublin, or mit)")
    parser.add_argument("-a", "--apnea_type",help="specify excerpt to use")
    parser.add_argument("-t", "--timesteps", help="specify length of sequence")
    parser.add_argument("-ep", "--epochs", help="specify number of epochs to train")
    parser.add_argument("-b", "--batch_size", help="specify batch size")
    parser.add_argument("-th", "--threshold", help="specify fraction between 0 and 1. if the predicted probability \
                                     is greater than this threshold then an apnea event is predicted.")
    args  = parser.parse_args()

    # store args 
    data        = args.data
    apnea_type  = args.apnea_type
    timesteps   = args.timesteps
    epochs      = args.epochs
    batch_size  = args.batch_size
    threshold   = args.threshold

    params = [epochs,batch_size,threshold]


    # for i in [2,3,5,6,7,8,9]:
    # apnea_type = f"{apnea_type}{i}"
    # params: <data> <apnea_type>, <timesteps>, <batch_size>, <threshold>
    # (program, data, timesteps, epochs, batch_size, threshold) = sys.argv
    print(f"********************** PROCESSING {apnea_type} *************************")
    print(f"\nPreprocessing {apnea_type} \n")
    subprocess.call(['python3', 'preprocessing.py', data, apnea_type, timesteps])
    
    print("\nTraining....\n")
    subprocess.call(['python3', 'rnn_train_only.py' ,data, apnea_type, timesteps, epochs, batch_size])
    
    print("\nTesting....\n")
    obj = subprocess.check_output(['python3', 'rnn_test_only.py', data, apnea_type, timesteps, batch_size, threshold],stderr=subprocess.STDOUT)
    print(obj)
    exit(0)
    with open(f'../meta/results.csv', "a") as out:
        writer = csv.DictWriter(out, fieldnames=['date','excerpt','precision','recall','f1','support','params'])
        writer.writerow({'date': datetime.datetime.today(), \
                        'excerpt':apnea_type, \
                        'precision':p,\
                        'recall':r,\
                        'f1':f1,\
                        'support':c,\
                        'params': params})
if __name__ == "__main__":
    main()