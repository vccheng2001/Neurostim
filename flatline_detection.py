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
FLATLINE_THRESHOLD = 0.01
WINDOW_SIZE = 100
def main():
    get_flatline_value()

def get_flatline_value():
# read unnormalized file
    file = f"{DATA_DIR}/{dataset}/preprocessing/excerpt{excerpt}/filtered_{sample_rate}hz.txt"
    df = pd.read_csv(file, delimiter=',')
    #plot
    df = df.iloc[:10000]
    df['Time1'] = df['Time']
    df.plot(x ='Time1', y='Value', kind = 'line')
    plt.show()


    # difference of values 1 sec apart 
    df['Diff'] = df['Value'].diff(10)
    # set to 0 if < THRESHOLD, else 1
    df['Diff'] = np.where(abs(df['Diff']) >= FLATLINE_THRESHOLD, 1, 0)
    # convert to binary string representation
    bin_list = df['Diff'].tolist()
    bin_str = ''.join(str(int(x)) for x in bin_list)

    # only mark as flatline if continuous flatline for 10 seconds)
    flatline_start_end_times, flatline_values = [], []
    print(len(df))
    print(len(bin_str))
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

        flatline_start_end_times.append({'start': start_time, 'end': end_time})
        flatline_values.append(avg_value)

    # avg flatline value across entire time series 
    avg_flatline_value = sum(flatline_values)/len(flatline_values)
    print(f"Avg detected flatline value: {avg_flatline_value}")
    return flatline_start_end_times, avg_flatline_value

if __name__ == "__main__":
    main()