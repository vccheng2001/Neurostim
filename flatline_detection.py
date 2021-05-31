import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import csv

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
DATASET = "mit"
EXCERPT = 37

SAMPLE_RATE= 10
FLATLINE_THRESHOLD = 0.01
WINDOW_SIZE = 100
SCALE_FACTOR = 100

base_path = f"{DATA_DIR}/{DATASET}/preprocessing/excerpt{EXCERPT}/filtered_{SAMPLE_RATE}hz" 

# path to unnormalized, normalized files 
unnorm_file = base_path + ".txt"
norm_file = base_path + f"_linear_{SCALE_FACTOR}.norm"

# output file with extracted flatline events
unnorm_out_file = base_path + f"_flatline_events.txt"
norm_out_file = base_path + f"_linear_{SCALE_FACTOR}_flatline_events.norm"

def main():
    # detect flatline events
    unnorm_flatline_times, _ = get_flatline_value(unnorm_file)
    norm_flatline_times, _   = get_flatline_value(norm_file, scale_factor=100)

    # writes detected flatline events to output file 
    output_flatline_files(unnorm_flatline_times, unnorm_out_file)
    output_flatline_files(norm_flatline_times, norm_out_file)



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
        for flatline_time in flatline_times:
            start_time, end_time = flatline_time
            writer.writerow({'OnSet': '%.3f' % start_time,
                            'Duration': '%.3f' % (end_time - start_time),
                            'Notation': 'FlatLine'})



'''
Detects flatline events by retrieving difference between signal values at adjacent
timesteps and binarizing into a string representation depending on whether the difference 
is less than a specified threshold.
0 means that the difference in signal value between two consecutive timesteps is negligible, 1 otherwise. 
We mark any repeating sequences of 0s spanning more than 10 seconds to be an apnea event. 
@param flatline_times: file with signal values sampled at <sample_rate> hz
                       file format is a csv with columns "Time", "Value"
       scale_factor:   scale factor used to normalize signal, default 1 if not specified
'''
def get_flatline_value(file, scale_factor=1):
    # read file
    df = pd.read_csv(file, delimiter=',')

    # comment out 
    df = df.iloc[:10000]

    # difference of values 1 sec apart (thus SAMPLE_RATE timesteps)
    df['Diff'] = df['Value'].diff(SAMPLE_RATE)
    # set to 0 if < THRESHOLD, else 1
    df['Diff'] = np.where(abs(df['Diff']) >= FLATLINE_THRESHOLD * scale_factor, 1, 0)
    
    
    # convert to binary string representation
    bin_list = df['Diff'].tolist()
    bin_str = ''.join(str(int(x)) for x in bin_list)

    # only mark as flatline if continuous flatline for 10 seconds
    flatline_times, flatline_values = [], []
    for x in re.finditer(r"(0)\1{" + re.escape(f"{SAMPLE_RATE * 10}") + r",}", bin_str):
        # print(f'Start time: {x.start()}, end_time: {x.end()}')
        start_idx = x.start()
        end_idx   = x.end()

        # get flatline start, end time intervals
        start_time = df.iloc[start_idx]['Time']         
        end_time = df.iloc[end_idx]['Time'] 

        # get avg flatline value
        avg_idx = int((start_idx+end_idx)/2)
        avg_value = df.iloc[avg_idx]['Value']

        flatline_times.append([start_time,end_time])
        flatline_values.append(avg_value)

    # avg flatline value across entire time series 
    flatline_value = sum(flatline_values)/len(flatline_values)
    print(f"Avg detected flatline value: {flatline_value}")


    # original plot
    df.plot(x ='Time', y='Value', kind = 'line')

    for l in flatline_times:
        plt.plot(l, [flatline_value, flatline_value], 'r-')
    plt.title(f"Avg detected flatline value: {flatline_value}")
    plt.show()
    return flatline_times, flatline_value

if __name__ == "__main__":
    main()