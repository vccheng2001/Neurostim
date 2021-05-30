import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re

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
dataset = "mit"
excerpt = 37
sample_rate = 10
SCALE_FACTOR = 100
FLATLINE_THRESHOLD = 0.01
WINDOW_SIZE = 100


def main():
    # unnormalized file
    unnorm_file = f"{DATA_DIR}/{dataset}/preprocessing/excerpt{excerpt}/filtered_{sample_rate}hz.txt" 
    unnorm_flatline_start_end_times, unnorm_flatline_value = get_flatline_value(unnorm_file)
    
    # normalized file
    norm_file = f"{DATA_DIR}/{dataset}/preprocessing/excerpt{excerpt}/filtered_{sample_rate}hz_linear_100.norm"
    norm_flatline_start_end_times, norm_flatline_value = get_flatline_value(norm_file)

def get_flatline_value(file):
    # read file
    df = pd.read_csv(file, delimiter=',')


    # comment out 
    df = df.iloc[:10000]

    # original plot
    # df.plot(x ='Time', y='Value', kind = 'line')
    # plt.show()

    # difference of values 1 sec apart 
    df['Diff'] = df['Value'].diff(10)
    # set to 0 if < THRESHOLD, else 1
    df['Diff'] = np.where(abs(df['Diff']) >= FLATLINE_THRESHOLD * SCALE_FACTOR, 1, 0)
    # convert to binary string representation
    bin_list = df['Diff'].tolist()
    bin_str = ''.join(str(int(x)) for x in bin_list)

    # only mark as flatline if continuous flatline for 10 seconds)
    flatline_start_end_times, flatline_values = [], []

    for x in re.finditer(r"(0)\1{100,}", bin_str):
        # print(f'Start time: {x.start()}, end_time: {x.end()}')
        start_idx = x.start()
        end_idx   = x.end()

        # get flatline start, end time intervals
        start_time = df.iloc[start_idx]['Time']         
        end_time = df.iloc[end_idx]['Time'] 

        # get avg flatline value
        avg_idx = int((start_idx+end_idx)/2)
        avg_value = df.iloc[avg_idx]['Value']

        flatline_start_end_times.append([start_time,end_time])
        flatline_values.append(avg_value)

    # avg flatline value across entire time series 
    flatline_value = sum(flatline_values)/len(flatline_values)
    print(f"Avg detected flatline value: {flatline_value}")


    # original plot
    df.plot(x ='Time', y='Value', kind = 'line')

    for l in flatline_start_end_times:
        plt.plot(l, [flatline_value, flatline_value], 'r-')
    plt.title(f"Avg detected flatline value: {flatline_value}")
    plt.show()
    return flatline_start_end_times, flatline_value

if __name__ == "__main__":
    main()