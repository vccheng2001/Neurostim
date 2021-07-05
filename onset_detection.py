import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import shutil
from datetime import datetime
import argparse 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import plotly
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px

from scipy.stats import zscore

''' Apnea event annotation '''

def main(args):
    # visualize original data
    fd = OnsetDetection(root_dir=".",
                           dataset=args.dataset,
                           apnea_type=args.apnea_type,
                           excerpt= args.excerpt,
                           sample_rate=args.sample_rate,
                           scale_factor=args.scale_factor)

    print('----------------Visualize original signal--------------------')

    fig = fd.visualize()
    print('----------------Onset Detection---------------------')

    # extract onset events
    onset_fig, onset_times, nononset_fig, nononset_times = fd.annotate_events(15, 0.1, 0.95)
    fig = make_subplots(rows=2, cols=1)

    for i in range(len(onset_fig['data'])):
        fig.add_trace(onset_fig['data'][i], row=1, col=1)
    for i in range(len(nononset_fig['data'])):
        fig.add_trace(nononset_fig['data'][i], row=2, col=1)

    fig.show()

    if args.full:
        print('----------------Output onset events--------------------')

        fd.output_apnea_files(onset_times, nononset_times)


class OnsetDetection():

    def __init__(self, root_dir, dataset, apnea_type, excerpt, sample_rate, scale_factor):

        self.dataset = dataset
        self.apnea_type  = apnea_type
        self.excerpt   = excerpt
        self.sample_rate = int(sample_rate)
        self.scale_factor = int(scale_factor)

        # directories 
        self.root_dir = root_dir
        self.data_dir = os.path.join(self.root_dir, "data")
        self.info_dir = os.path.join(self.root_dir, "info")

        self.base_path = f"{self.data_dir}/{self.dataset}/preprocessing/excerpt{self.excerpt}/{self.dataset}_{self.apnea_type}_ex{self.excerpt}_sr{self.sample_rate}"
        # path to unnormalized, normalized files 
        self.in_file     = f"{self.base_path}_sc{self.scale_factor}.csv"
        # output file with extracted onset events
        self.out_file = f"{self.base_path}_sc{self.scale_factor}_onset_events.txt"
        # pos/neg sequence files 
        self.sequence_dir = f"{self.data_dir}/{self.dataset}/postprocessing/excerpt{self.excerpt}/"
        # default parameters 
        self.seconds_before_apnea = 10
        self.seconds_after_apnea = 10
        self.window_size_seconds = 10 # sliding window # seconds
        

    ''' Generate figures '''
    def visualize(self):
        # plot original data 
        df = pd.read_csv(self.in_file, delimiter=',')
        pd.options.plotting.backend = "plotly"
        fig = df.plot(x='Time', y="Value",  width=1700, height=600)
        fig.update_traces(line=dict(color="gray", width=0.5))
        return fig

    '''
    Writes detected onset events to output file 
    @param onset_times: list of [start, end] times
        out_file: file to write to 
    '''
    def output_apnea_files(self, onset_times, nononset_times):
        with open(self.out_file, 'w', newline='\n') as out:
            fieldnames = ["OnSet", "Duration", "Notation"]
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()

            # write each detected onset event as a row
            for start_time, end_time in onset_times:
                writer.writerow({'OnSet': '%.3f' % start_time,
                                'Duration': '%.3f' % (end_time - start_time)})

        # initialize directories 
        init_dir(self.sequence_dir)
        pos_dir = self.sequence_dir + "positive/"
        neg_dir = self.sequence_dir + "negative/"
        init_dir(pos_dir)
        init_dir(neg_dir)

        # write positive sequences, one file for each onset apnea event
        df = pd.read_csv(self.in_file, delimiter=',')
        
        # apnea start time
        for start_time, end_time in onset_times:

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
        for start_time, end_time in nononset_times: 
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
    Detects onset events using string pattern matching algorithm
    @param file: file with signal values sampled at <sample_rate> hz
                        file format is a csv with columns "Time", "Value"
        scale_factor:   scale factor used to normalize signal, default 1 if not specified
        norm: indicates whether using unnormalized or normalized file
    '''
    def annotate_events(self, onset_threshold=15, onset_thresh_frac=0.1, nononset_thresh_frac=0.9, norm=True):
        # read file
        df = pd.read_csv(self.in_file, delimiter=',')

        # uncomment if using subset of signal 
        # df = df.iloc[20000*self.sample_rate:30000*self.sample_rate]

        df["Value"] = zscore(df["Value"])
        df['Binary_Value'] = np.where(abs(df['Value']) >= (onset_threshold * self.scale_factor), 1, 0)
        df["Rolling_Mean"] = df["Value"].rolling(self.sample_rate*10).mean()

        # df["EWM"] = df["Value"].ewm(self.sample_rate).mean()
        # df['Diff'] = df['Value'].diff(self.sample_rate)
        # df['Binary_Diff'] = np.where(abs(df['Diff']) >= (onset_threshold * self.scale_factor), 1, 0)

    
            
        print('--------Plotting orig, diff------------')
        fig = make_subplots(rows=3, cols=1)
        # plot original time series 
        fig_orig = df.plot(x ='Time', y='Value', kind = 'line')
        fig_rolling_mean = df.plot(x ='Time', y='Rolling_Mean', kind = 'line')
        fig_binary_value = df.plot(x ='Time', y='Binary_Value', kind = 'line')

        fig.add_trace(fig_orig['data'][0], row=1, col=1)
        fig.add_trace(fig_rolling_mean['data'][0], row=2, col=1)
        fig.add_trace(fig_binary_value['data'][0], row=3, col=1)

        fig.show()

        # convert to binary list 
        bin_list = df['Binary_Value'].tolist()
        bin_str = ''.join(str(int(x)) for x in bin_list)
        n = len(bin_list)                                         # length of time series 
        k = int(self.sample_rate * self.window_size_seconds)      # window size (number of timesteps)
        nononset_times, onset_times, onset_values = [], [], []     
       

        '''----------------Extracting nononset events----------------------'''
        import re
        for x in re.finditer(r"(0)\1{" + re.escape(f"{int(self.sample_rate)* 10}") + r",}", bin_str):
            # print(f'Start time: {x.start()}, end_time: {x.end()}')

            # Flatline duration
            start_idx = x.start() 
            end_idx   = x.end()

            # get onset start, end time intervals
            start_time = df.iloc[start_idx-(self.seconds_before_apnea * self.sample_rate)]['Time']        
            end_time = df.iloc[start_idx + (self.seconds_after_apnea * self.sample_rate)]['Time'] 

            # get average flatline value from center of onset event 
            avg_idx = int((start_idx+end_idx)/2)
            avg_value = df.iloc[avg_idx]['Value']

            # append as onset event 
            onset_times.append([start_time,end_time])
            onset_values.append(avg_value)

        try:
            # avg onset value across entire time series 
            onset_value = np.mean(onset_values)

        except ZeroDivisionError:
            print("No onset events found!")
            exit(-1)


        pd.options.plotting.backend = "plotly"
    
        onset_fig = px.line(df, x="Time", y="Value", title='Extracted onset events (red)')
        onset_fig.update_traces(line=dict(color="gray", width=0.5))


        # get nononset
        nononset_times = []
        total_sec = self.seconds_before_apnea + self.seconds_after_apnea
        prev_end_time = None
        for next_start_time, next_end_time in onset_times:
            if prev_end_time is None: 
                prev_end_time = next_end_time
                continue

            num_segments = (next_start_time - prev_end_time) // total_sec 
            nononset_start = prev_end_time
            for i in range(int(num_segments)):
                nononset_times.append([nononset_start, nononset_start + total_sec])
                nononset_start += total_sec
            prev_end_time = next_end_time
                
     
        ''''--------------------Plot detected onset vents----------------------'''

        pd.options.plotting.backend = "plotly"
    
        onset_fig = px.line(df, x="Time", y="Value", title='Extracted onset events (red)')
        onset_fig.update_traces(line=dict(color="gray", width=0.5))

        for ft in onset_times:
            onset_fig.add_trace(
                go.Scatter(
                    x=ft,
                    y=[onset_value, onset_value],
                    mode="lines",
                    line=go.scatter.Line(color="red", width=8),
                    showlegend=True
                )
            )

        onset_fig.update_layout(
                autosize=False,
                width=1700,
                height=600,)

        ''' -----------------Plot detected nononset events--------------------'''
        nononset_fig = px.line(df, x="Time", y="Value", title='Extracted nononset events (green)')
        nononset_fig.update_traces(line=dict(color="gray", width=0.5))
        for nft in nononset_times:
            nononset_fig.add_trace(
                go.Scatter(
                    x=nft,
                    y=[onset_value, onset_value],
                    mode="lines",
                    line=go.scatter.Line(color="green", width=8),
                    showlegend=True
                )
            )

        nononset_fig.update_layout(
                autosize=False,
                width=1700,
                height=600,)


        print(f"Extracted {len(onset_times)} onset events")
        print(f"Extracted {len(nononset_times)} non-onset events")

        
        self.onset_times = onset_times
        self.nononset_times = nononset_times
        return onset_fig, onset_times, nononset_fig, nononset_times


''' Helper function to create directory '''
def init_dir(path): 
    if os.path.isdir(path): shutil.rmtree(path)
    if not os.path.isdir(path):
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
    main(args)
