import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import shutil
from datetime import datetime
import argparse 


# plot
import plotly
from plotly.offline import plot

import plotly.graph_objects as go
import plotly.express as px

''' Apnea event annotation '''

import matplotlib
matplotlib.use('agg')

class FlatlineDetection():

    def __init__(self, root_dir, dataset, apnea_type, excerpt, sample_rate, scale_factor):

        print('in init......')
        print('dataset', dataset)
        print('sr', sample_rate)
        self.dataset = dataset
        self.apnea_type  = apnea_type
        self.excerpt   = int(excerpt)
        self.sample_rate = int(sample_rate)
        self.scale_factor = int(scale_factor)

        # directories 
        self.root_dir = root_dir
        self.data_dir = os.path.join(self.root_dir, "data")
        self.info_dir = os.path.join(self.root_dir, "info")

        self.base_path = f"{self.data_dir}/{self.dataset}/preprocessing/excerpt{self.excerpt}/{self.dataset}_{self.apnea_type}_ex{self.excerpt}_sr{self.sample_rate}"
        # path to unnormalized, normalized files 
        self.in_file     = f"{self.base_path}_sc{self.scale_factor}.txt"
        # output file with extracted flatline events
        self.out_file = f"{self.base_path}_sc{self.scale_factor}_flatline_events.txt"
        # pos/neg sequence files 
        self.sequence_dir = f"{self.data_dir}/{self.dataset}/postprocessing/excerpt{self.excerpt}/"
        # default parameters 
        self.seconds_before_apnea = 10
        self.seconds_after_apnea = 5
        self.window_size_seconds = 10 # sliding window # seconds
        

    ''' Generate figures '''
    def visualize(self):
        # plot original data 
        df = pd.read_csv(self.in_file, delimiter=',')
        df = df.iloc[:10000]
        pd.options.plotting.backend = "plotly"
        fig = df.plot(x='Time', y="Value",  width=1700, height=600)

        # plot
        return fig

    '''
    Writes detected flatline events to output file 
    @param flatline_times: list of [start, end] times
        out_file: file to write to 
    '''
    def output_apnea_files(self, flatline_times, nonflatline_times):
        with open(self.out_file, 'w', newline='\n') as out:
            fieldnames = ["OnSet", "Duration", "Notation"]
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()

            # write each detected flatline event as a row
            for start_time, end_time in flatline_times:
                writer.writerow({'OnSet': '%.3f' % start_time,
                                'Duration': '%.3f' % (end_time - start_time),
                                'Notation': 'FlatLine'})

        # initialize directories 
        init_dir(self.sequence_dir)
        pos_dir = self.sequence_dir + "positive/"
        neg_dir = self.sequence_dir + "negative/"
        init_dir(pos_dir)
        init_dir(neg_dir)

        # write positive sequences, one file for each flatline apnea event
        df = pd.read_csv(self.in_file, delimiter=',')
        
        # apnea start time
        for start_time, end_time  in flatline_times:

            pos_out_file = f'{start_time}.txt'
            try:
                # slice from <SECONDS_BEFORE_APNEA> sec before apnea to <SECONDS_AFTER_APNEA> sec after
    
                start_idx = df.index[df["Time"] == round(start_time -  self.seconds_before_apnea, 3)][0]
                end_idx =   df.index[df["Time"] == round(start_time +  self.seconds_after_apnea, 3)][0]

                df.iloc[start_idx:end_idx,  df.columns.get_loc('Value')].to_csv(pos_dir + pos_out_file,\
                                                    index=False, header=False, float_format='%.3f')
            except:
                continue
        
        # write negative sequences 
        for start_time, end_time in nonflatline_times: 
            neg_out_file = f'{start_time}.txt' 

            try:
                # slice for <SECONDS_BEFORE_APNEA + SECONDS_AFTER_APNEA> seconds 
                start_idx = df.index[df["Time"] == round(start_time, 3)][0]
                end_idx = df.index[df["Time"] == round(start_time + (self.seconds_before_apnea + self.seconds_after_apnea), 3)][0]

                df.iloc[start_idx:end_idx,  df.columns.get_loc('Value')].to_csv(neg_dir + neg_out_file,\
                                                index=False, header=False, float_format='%.3f')
            except:
                # check if out of bounds 
                continue

  
    '''
    Detects flatline events using string pattern matching algorithm
    @param file: file with signal values sampled at <sample_rate> hz
                        file format is a csv with columns "Time", "Value"
        scale_factor:   scale factor used to normalize signal, default 1 if not specified
        norm: indicates whether using unnormalized or normalized file
    '''
    def annotate_events(self, flatline_threshold=15, flatline_thresh_frac=0.1, nonflatline_thresh_frac=0.9, norm=True):
        # read file
        df = pd.read_csv(self.in_file, delimiter=',')

        # uncomment if using subset of signal 
        # df = df.iloc[:10000]

        # plot original time series 
        df.plot(x ='Time', y='Value', kind = 'line')
        # plt.show()

        
        # difference of values 1 sec apart (thus SAMPLE_RATE timesteps)
        df['Diff'] = df['Value'].diff(self.sample_rate)
        # set to 0 if < THRESHOLD, else 1
        df['Binary_Diff'] = np.where(abs(df['Diff']) >= (flatline_threshold * self.scale_factor), 1, 0)
        # convert to binary list 
        bin_list = df['Binary_Diff'].tolist()


        n = len(bin_list)                                         # length of time series 
        k = int(self.sample_rate) * self.window_size_seconds      # window size (number of timesteps)
        nonflatline_times, flatline_times, flatline_values = [], [], []     
        flatline_ratio = int(k * flatline_thresh_frac)           # max % of 0s in window to count as flatline
        nonflatline_ratio = int(k * nonflatline_thresh_frac)     # min % of 0s in window to count as nonflatline 


        '''----------------Extracting nonflatline events----------------------'''

        # sliding window over time series 
        i = 0
        while (i + k) < n: 
            window_sum = sum(bin_list[i:i+k])

            # if detect flatline 
            if window_sum < flatline_ratio:
                start_idx = i
                end_idx = i+k
                # get flatline start, end time intervals
                start_time = df.iloc[start_idx]['Time']        
                end_time = df.iloc[end_idx]['Time'] 

                # get average flatline value from ceneter of flatline event 
                avg_idx = int((start_idx+end_idx)/2)
                avg_value = df.iloc[avg_idx]['Value']

                # append as flatline event 
                flatline_times.append([start_time,end_time])
                flatline_values.append(avg_value)
                i += k
            # if no flatline, advance to next timestep 
            else: i += 1

        try:
            # avg flatline value across entire time series 
            flatline_value = sum(flatline_values)/len(flatline_values)
        except ZeroDivisionError:
            print("No flatline events found!")
            exit(-1)


        '''----------------Extracting nonflatline events----------------------'''
        i = 0
        while (i + k) < n:
            window_sum = sum(bin_list[i:i+k])
            if window_sum > nonflatline_ratio:
                start_idx = i
                end_idx = i+k 
                # get flatline start, end time intervals
                start_time = df.iloc[start_idx-1]['Time']        
                end_time = df.iloc[end_idx-1]['Time']  
                nonflatline_times.append([start_time,end_time])
                i += k
            else: i += 1


        ''''--------------------Plot detected flatline vents----------------------'''

        pd.options.plotting.backend = "plotly"
    
        flatline_fig = px.line(df, x="Time", y="Value", title='Extracted flatline events (red)')
        for ft in flatline_times:
            flatline_fig.add_trace(
                go.Scatter(
                    x=ft,
                    y=[flatline_value, flatline_value],
                    mode="lines",
                    line=go.scatter.Line(color="red", width=5),
                    showlegend=True
                )
            )

        flatline_fig.update_layout(
                autosize=False,
                width=1700,
                height=600,)

        ''' -----------------Plot detected nonflatline events--------------------'''
        nonflatline_fig = px.line(df, x="Time", y="Value", title='Extracted nonflatline events (green)')
        for nft in nonflatline_times:
            nonflatline_fig.add_trace(
                go.Scatter(
                    x=nft,
                    y=[flatline_value, flatline_value],
                    mode="lines",
                    line=go.scatter.Line(color="green", width=5),
                    showlegend=True
                )
            )

        nonflatline_fig.update_layout(
                autosize=False,
                width=1700,
                height=600,)


        # print(f"Extracted {len(flatline_times)} flatline events")
        # print(f"Extracted {len(nonflatline_times)} flatline events")

        
        self.flatline_times = flatline_times
        self.nonflatline_times = nonflatline_times
        return flatline_fig, flatline_times, nonflatline_fig, nonflatline_times


''' Helper function to create directory '''
def init_dir(path): 
    if os.path.isdir(path): shutil.rmtree(path)
    if not os.path.isdir(path):
        os.mkdir(path)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-d", "--dataset",    default="dreams", help="dataset (dreams, dublin, or mit)")
#     parser.add_argument("-a", "--apnea_type", default="osa",    help="type of apnea (osa, osahs, or all)")
#     parser.add_argument("-ex","--excerpt",    default=1,        help="excerpt number to use")
#     parser.add_argument("-sr","--sample_rate",    default=10,        help="number of samples per second")
#     parser.add_argument("-sc","--scale_factor",    default=10,        help="scale factor for normalization")

#     # parse args 
#     args = parser.parse_args()

#     # store args 
#     DATASET   = args.dataset
#     APNEA_TYPE  = args.apnea_type
#     EXCERPT   = int(args.excerpt)
#     SAMPLE_RATE = int(args.sample_rate)
#     SCALE_FACTOR = int(args.scale_factor)

#     base_path = f"{DATA_DIR}/{DATASET}/preprocessing/excerpt{EXCERPT}/filtered_{SAMPLE_RATE}hz" 

#     # path to unnormalized, normalized files 
#     file = base_path + f"_linear_{SCALE_FACTOR}.norm"

#     # output file with extracted flatline events
#     out_file = base_path + f"_linear_{SCALE_FACTOR}_flatline_events.norm"

#     # pos/neg sequence files 
#     sequence_dir = f"{DATA_DIR}/{DATASET}/postprocessing/excerpt{EXCERPT}/"

#     main()