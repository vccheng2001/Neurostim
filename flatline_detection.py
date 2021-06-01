import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import csv
import shutil
from datetime import datetime
import argparse 
'''
Program to annotate apnea events
1. Identify flatline areas and identify value for flatline
2. Annotate flatline apnea events 
3. Output to positive, negative sequences 
'''

pd.set_option("display.max_rows", 2000, "display.max_columns", 2000)

# directories 
ROOT_DIR = os.getcwd() 
DATA_DIR = os.path.join(ROOT_DIR, "data")
INFO_DIR = os.path.join(ROOT_DIR, "info")


# DATASET = "mit"
# APNEA_TYPE = apnea_type
# EXCERPT = 37
# SAMPLE_RATE= 10
# SCALE_FACTOR = 100

# preset parameters 
FLATLINE_THRESHOLD = 0.1
SECONDS_BEFORE_APNEA = 8
SECONDS_AFTER_APNEA = 5

def main():
    # detect flatline events
    # unnorm_flatline_times, unnorm_nonflatline_times = annotate_signal(unnorm_file)
    norm_flatline_times, norm_nonflatline_times   = annotate_signal(norm_file, scale_factor=SCALE_FACTOR, norm=True)

    # writes detected flatline events to output file 
    # output_flatline_files(unnorm_flatline_times, unnorm_out_file)
    output_flatline_files(norm_flatline_times, norm_out_file)


    # create positive, negative sequence files for training 
    # output_pos_neg_seq(sequence_dir, unnorm_file, unnorm_flatline_times, unnorm_nonflatline_times)
    output_pos_neg_seq(sequence_dir, norm_file, norm_flatline_times, norm_nonflatline_times)


'''
Creates positive, negative sequences
@param sequence_dir: directory to store pos/neg sequences
       flatline_times: list of [start, end] times 
       file: csv file containing time, signal value
'''
def output_pos_neg_seq(sequence_dir, file, flatline_times, nonflatline_times): 
    # initialize directories 
    init_dir(sequence_dir)
    pos_dir = sequence_dir + "positive/"
    neg_dir = sequence_dir + "negative/"
    init_dir(pos_dir)
    init_dir(neg_dir)

    # write positive sequences, one file for each flatline apnea event
    df = pd.read_csv(file, delimiter=',')
    for start_time, end_time  in flatline_times:

        out_file = f'{start_time}.txt'
        # get starting, ending indices to slice 
        start_idx = df.index[df["Time"] == round(start_time - SAMPLE_RATE * SECONDS_BEFORE_APNEA, 3)][0]
        end_idx =   df.index[df["Time"] == round(start_time +  SAMPLE_RATE * SECONDS_AFTER_APNEA, 3)][0]
        # print(f'Creating positive sequence from timestep {start_idx} to {end_idx} ')

        # slice from <SECONDS_BEFORE_APNEA> sec before apnea to <SECONDS_AFTER_APNEA> sec after
        # write to csv files
        df.iloc[start_idx:end_idx,  df.columns.get_loc('Value')].to_csv(pos_dir + out_file,\
                                             index=False, header=False, float_format='%.3f')

    # write negative sequences 

    for start_time, end_time in nonflatline_times: 
        out_file = f'{start_time}.txt' 

        start_idx = df.index[df["Time"] == round(start_time, 3)]

        try:
            end_idx = df.index[df["Time"] == round(end_time, 3)]
        except:
            # check if out of bounds 
            continue
        df.iloc[start_idx[0]:end_idx[0],  df.columns.get_loc('Value')].to_csv(neg_dir + out_file,\
                                             index=False, header=False, float_format='%.3f')
        # print(f'Creating negative sequence from timestep {start_idx} to {end_idx} ')





'''
Writes detected flatline events to output file 
@param flatline_times: list of [start, end] times
       out_file: file to write to 
'''
def output_flatline_files(flatline_times, out_file):
    print('out file', out_file)
    with open(out_file, 'w', newline='\n') as out:
        fieldnames = ["OnSet", "Duration", "Notation"]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        # write each detected flatline event as a row
        for start_time, end_time in flatline_times:
            writer.writerow({'OnSet': '%.3f' % start_time,
                            'Duration': '%.3f' % (end_time - start_time),
                            'Notation': 'FlatLine'})

'''
Gets non-flatline regions in signal. Used to make negative sequences
@param flatline_times: file with signal values sampled at <sample_rate> hz
'''
def get_nonflatlines(flatline_times):
    nonflatline_times = []
    TOTAL_SEC = SECONDS_BEFORE_APNEA + SECONDS_AFTER_APNEA
    for start_time, end_time in flatline_times:
        if end_time - start_time >= TOTAL_SEC:
            nonflatline_times.append([end_time, end_time + TOTAL_SEC])
    return nonflatline_times


'''
Detects flatline events by retrieving difference between signal values at adjacent
timesteps and binarizing into a string representation depending on whether the difference 
is less than a specified threshold.
0 means that the difference in signal value between two consecutive timesteps is negligible, 1 otherwise. 
We mark any repeating sequences of 0s spanning more than 10 seconds to be an apnea event. 
@param files file with signal values sampled at <sample_rate> hz
                       file format is a csv with columns "Time", "Value"
       scale_factor:   scale factor used to normalize signal, default 1 if not specified
       norm: indicates whether using unnormalized or normalized file
'''
def annotate_signal(file, scale_factor=1, norm=False):
    # read file
    df = pd.read_csv(file, delimiter=',')

    # comment out 
    # df = df.iloc[5000:15000]

    
    # difference of values 1 sec apart (thus SAMPLE_RATE timesteps)
    df['Diff'] = df['Value'].diff(SAMPLE_RATE)
    # set to 0 if < THRESHOLD, else 1


    df['Binary_Diff'] = np.where(abs(df['Diff']) >= SCALE_FACTOR*100, 1, 0)

    # convert to binary string representation
    bin_list = df['Binary_Diff'].tolist()
    bin_str = ''.join(str(int(x)) for x in bin_list)

    # # plot orig
    # df.plot(x ='Time', y='Value', kind = 'line')
    # plt.show()

    # only mark as flatline if continuous flatline for 10 seconds
    flatline_times, flatline_values = [], []

    for x in re.finditer(r"(0)\1{" + re.escape(f"{int(SAMPLE_RATE)* 10}") + r",}", bin_str):
        # print(f'Start time: {x.start()}, end_time: {x.end()}')
        start_idx = x.start()
        end_idx   = x.end()
        # get flatline start, end time intervals
        start_time = df.iloc[start_idx-1]['Time']    
             
        end_time = df.iloc[end_idx-1]['Time'] 
        # get avg flatline value
        avg_idx = int((start_idx+end_idx)/2)
        avg_value = df.iloc[avg_idx]['Value']

        flatline_times.append([start_time,end_time])
        flatline_values.append(avg_value)


    try:
        # avg flatline value across entire time series 
        flatline_value = sum(flatline_values)/len(flatline_values)
    except ZeroDivisionError:
        print("No flatline events found!")
        exit(-1)

    # get non-flatline times 
    nonflatline_times = get_nonflatlines(flatline_times)
 

    # original plot
    df.plot(x ='Time', y='Value', kind = 'line')

    for ft in flatline_times:
        plt.plot(ft, [flatline_value, flatline_value], 'r-')
    for nft in nonflatline_times:
        plt.plot(nft, [flatline_value, flatline_value], 'y-')

    if norm: 
        plt.title(f"Avg detected flatline value (NORMALIZED): {flatline_value}")
    else:
        plt.title(f"Avg detected flatline value (UNNORMALIZED): {flatline_value}")
 
    # plt.show()
    return flatline_times, nonflatline_times

''' Helper function to create directory '''
def init_dir(path): 
    if os.path.isdir(path): shutil.rmtree(path)
    if not os.path.isdir(path):
        # # print("Making directory.... " + path)
        os.mkdir(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",    default="dreams", help="dataset (dreams, dublin, or mit)")
    parser.add_argument("-a", "--apnea_type", default="osa",    help="type of apnea (osa, osahs, or all)")
    parser.add_argument("-ex","--excerpt",    default=1,        help="excerpt number to use")
    parser.add_argument("-sr","--sample_rate",    default=10,        help="number of samples per second")
    parser.add_argument("-sc","--scale_factor",    default=10,        help="scale factor for normalization")

    # parse args 
    args = parser.parse_args()

    # print(args)
    # store args 
    DATASET   = args.dataset
    APNEA_TYPE  = args.apnea_type
    EXCERPT   = int(args.excerpt)
    SAMPLE_RATE = int(args.sample_rate)
    SCALE_FACTOR = int(args.scale_factor)

    base_path = f"{DATA_DIR}/{DATASET}/preprocessing/excerpt{EXCERPT}/filtered_{SAMPLE_RATE}hz" 

    # path to unnormalized, normalized files 
    unnorm_file = base_path + ".txt"
    norm_file = base_path + f"_linear_{SCALE_FACTOR}.norm"

    # output file with extracted flatline events
    unnorm_out_file = base_path + f"_flatline_events.txt"
    norm_out_file = base_path + f"_linear_{SCALE_FACTOR}_flatline_events.norm"

    # pos/neg sequence files 
    sequence_dir = f"{DATA_DIR}/{DATASET}/postprocessing/excerpt{EXCERPT}/"

    main()