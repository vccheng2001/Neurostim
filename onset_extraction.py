# Util
import os
import pandas as pd
import numpy as np
import csv
import shutil
from datetime import datetime
import argparse 
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler


import wandb
from wandb import log
# Torch
import torch 

# Plot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotly
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
pd.options.plotting.backend = "plotly"

# Stats 
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression

# # Pandas config 
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 150)

''' Onset events extraction module '''

def main(cfg):

    oe = OnsetExtraction(cfg=cfg)
    oe.orig_fig = oe.visualize(time_field="Time", value_field="Value")

    if cfg.normalize:
                        
        # Normalize 
        oe.normalize(slope_threshold=float(cfg.slope_threshold),
                        scale_factor_high=int(cfg.scale_factor_high),
                        scale_factor_low=int(cfg.scale_factor_low))
        
    oe.plot_extracted_events()

    # Extract onset, non-onset  events
    oe.extract_onset_events()

    # Save positive/negative sequences to files
    oe.write_output_files()



class OnsetExtraction():

    def __init__(self, cfg=None):

        print('------Initializing OnsetExtraction module-------')
        self.logger = cfg.logger

        self.dataset = cfg.dataset
        self.apnea_type = cfg.apnea_type
        self.excerpt   = cfg.excerpt
        self.sample_rate = int(cfg.sample_rate)

        self.slope_threshold = float(cfg.slope_threshold)
        self.scale_factor_high = int(cfg.scale_factor_high)
        self.scale_factor_low = int(cfg.scale_factor_high)


        # define directories 
        self.root_dir = cfg.root_dir
        self.data_dir = os.path.join(self.root_dir, "data")
        self.info_dir = os.path.join(self.root_dir, "info")
        self.negative_dir = cfg.negative_dir
        self.positive_dir = cfg.positive_dir

        # raw input file with columns [Time, Value]
        self.base_path = f"{self.data_dir}/{self.dataset}/preprocessing/excerpt{self.excerpt}/{self.dataset}_{self.apnea_type}_ex{self.excerpt}_sr{self.sample_rate}"
        self.in_file = f"{self.base_path}_sc1.csv"
        self.df = pd.read_csv(self.in_file, delimiter=',')

        self.df["Orig_Value"] = self.df["Value"]

        # output file to list all extracted onset events
        self.out_file = f"{self.base_path}_sc1_onset_events.txt"
        # pos/neg sequence files 
        self.sequence_dir = f"{self.data_dir}/{self.dataset}/postprocessing/excerpt{self.excerpt}/"
        # default parameters 
        self.seconds_before_apnea = int(cfg.seconds_before_apnea)
        self.seconds_after_apnea = int(cfg.seconds_after_apnea)
        
        # original fig
        self.orig_fig = self.visualize()

 
    ''' Visualize original signal '''
    def visualize(self, time_field="Time", value_field="Value"):
        fig = self.df.plot(x=time_field, y=value_field,  width=1700, height=600)
        fig.update_traces(line=dict(color="gray", width=0.5))
        return fig

    '''
    Detects onset events using string pattern matching algorithm
    @param file: file with signal values sampled at <sample_rate> hz
                        file format is a csv with columns "Time", "Value"
    '''
    def normalize(self, slope_threshold=10, 
                        scale_factor_high=10, 
                        scale_factor_low=0.1):

        print('-----Normalizing-------')
        self.slope_threshold = slope_threshold
        self.scale_factor_high = scale_factor_high
        self.scale_factor_low = scale_factor_low
        
        # Calculated slope across every second
        self.df["Slope"] = self.df["Value"].rolling(window=8, min_periods=1).apply(lambda x: (x[-1] - x[0]) / 2,  raw=True)
        # Get scale factor based on calculated slope
        self.df["Scale_Factor"] = np.where(abs(self.df["Slope"]) > self.slope_threshold, self.scale_factor_high, self.scale_factor_low)
        # Scale nonlinearly
        self.df["Value"] *= self.df["Scale_Factor"] # (self.df["Value"] ** 2) * np.sign(self.df["Value"])#*  self.df["Scale_Factor"]
        # Return normalized signal
        self.df["Value"] = zscore(self.df["Value"])
        return self.df




    ''' Extracts onset events'''
    def extract_onset_events(self):


        self.onset_times = []
        self.flatline_times = []
        flatline_values = []


        self.df["Variance"] = self.df["Value"].rolling(window=80, min_periods=1).apply(lambda x: np.var(x), raw=True)
        self.df['Binary_Var'] = np.where(abs(self.df['Variance']) >= 0.1, 1, 0)
        bin_list = self.df['Binary_Var'].tolist()
        bin_str = ''.join(str(int(x)) for x in bin_list)        

        # Search positive sequences at least 10 seconds
        for x in re.finditer(r"11111111(0)\1{" + re.escape(f"{int(self.sample_rate)* 10}") + r",}", bin_str):

            start_idx = x.start()
            end_idx = x.end()

            # get onset start, end time intervals
            if start_idx - (self.seconds_before_apnea * self.sample_rate) < 0:
                continue

            # Start of Onset:  10 seconds before flatline
            # End of Onset: 5 seconds after after flatline
            start_time = self.df.iloc[start_idx-(self.seconds_before_apnea * self.sample_rate)]['Time']        
            end_time = self.df.iloc[start_idx + (self.seconds_after_apnea * self.sample_rate)]['Time'] 

            # Start, End of flatline
            start_flatline = self.df.iloc[start_idx+self.sample_rate]["Time"]
            end_flatline = self.df.iloc[end_idx-1]["Time"]

            # get average flatline value from center of onset event 
            avg_flatline_idx = int((start_idx+end_idx)/2)
            avg_flatline_value = self.df.iloc[avg_flatline_idx]['Value'] # VIVI

            # append onset event 
            self.onset_times.append([start_time,end_time])

            # append flatline event
            # NOTE: GOES FROM START ONSET ---> END OF FLATLINE
            self.flatline_times.append([start_time, end_flatline])


            # append avg flatline value
            flatline_values.append(avg_flatline_value)

        self.flatline_value = np.mean(flatline_values)


        ''' Extract non-onset (regular breathing) events'''

        self.nononset_times = []

        self.df["Variance"] = self.df["Value"].rolling(window=80, min_periods=1).apply(lambda x: np.var(x), raw=True)
        self.df['Binary_Var'] = np.where(abs(self.df['Variance']) >= 0.25, 1, 0)
        bin_list = self.df['Binary_Var'].tolist()
        bin_str = ''.join(str(int(x)) for x in bin_list)


        # Search for high variance events
        for x in re.finditer(r"(1)\1{" + re.escape(f"{int(self.sample_rate) * int(self.seconds_before_apnea) + int(self.seconds_after_apnea)}") + r",}", bin_str):

            start_idx = x.start()
            end_idx = x.end()

            # get onset start, end time intervals
            if start_idx - (self.seconds_before_apnea * self.sample_rate) < 0:
                continue
            start_time = self.df.iloc[start_idx-1]['Time']        
            end_time = self.df.iloc[end_idx-1]['Time'] 
            # append non-onset event 
            self.nononset_times.append([start_time,end_time])


        num_pos_files = len(self.onset_times)
        num_neg_files = len(self.nononset_times)
        print(f"Extracted {num_pos_files} onset events")
        print(f"Extracted {num_neg_files } non-onset events")


    # Find flatline value of df[<col>]
    def find_flatline_value(self, col):
        
        self.df["Variance"] = self.df[col].rolling(window=80, min_periods=1).apply(lambda x: np.var(x), raw=True)
        self.df['Binary_Var'] = np.where(abs(self.df['Variance']) >= 0.5, 1, 0)
        bin_list = self.df['Binary_Var'].tolist()
        bin_str = ''.join(str(int(x)) for x in bin_list)

        '''----------------Extracting onset events----------------------'''
        flatline_values = []
        for x in re.finditer(r"(0)\1{" + re.escape(f"{int(self.sample_rate)* 10}") + r",}", bin_str):
            # print(f'Start time: {x.start()}, end_time: {x.end()}', x.end() - x.start())
            # Flatline duration
            start_idx = x.start() 
            end_idx   = x.end()
            avg_flatline_idx = int((start_idx+end_idx)/2)
            avg_flatline_value = self.df.iloc[avg_flatline_idx][col]
            flatline_values.append(avg_flatline_value)
            # break
        try:
            flatline_value = np.mean(flatline_values)
        except ZeroDivisionError:
            print("No flatline values found!")
            exit(-1)
        return flatline_value

    
    def plot_extracted_events(self):
        ''''--------------------Plot extracted onset vents----------------------'''
    
        self.orig_fig.update_layout(
                autosize=False,
                width=1700,
                height=600,)


        self.onset_fig = px.line(self.df, x="Time", y="Value", title='Extracted onset events (red)')
        self.onset_fig.update_traces(line=dict(color="gray", width=0.5))

        for ft in self.onset_times:
            self.onset_fig.add_trace(
                go.Scatter(
                    x=ft,
                    y=[self.flatline_value, self.flatline_value],
                    mode="lines",
                    line=go.scatter.Line(color="red", width=8),
                    showlegend=True
                )
            )

        self.onset_fig.update_layout(
                autosize=False,
                width=1700,
                height=600,)

        ''' -----------------Plot extracted non-onset events--------------------'''
        self.nononset_fig = px.line(self.df, x="Time", y="Value", title='Extracted nononset events (green)')
        self.nononset_fig.update_traces(line=dict(color="gray", width=0.5))
        for nft in self.nononset_times:
            self.nononset_fig.add_trace(
                go.Scatter(
                    x=nft,
                    y=[self.flatline_value, self.flatline_value],
                    mode="lines",
                    line=go.scatter.Line(color="green", width=8),
                    showlegend=True
                )
            )

        self.nononset_fig.update_layout(
                autosize=False,
                width=1700,
                height=600,)

        fig = make_subplots(rows=3, cols=1)

        for i in range(len(self.orig_fig['data'])):
            fig.add_trace(self.orig_fig['data'][i], row=1, col=1)
        for i in range(len(self.onset_fig['data'])):
            fig.add_trace(self.onset_fig['data'][i], row=2, col=1)
        for i in range(len(self.nononset_fig['data'])):
            fig.add_trace(self.nononset_fig['data'][i], row=3, col=1)

        # if self.logger:
        #     wandb.log({"extracted_events": fig})
        fig.update_layout(title=f'{self.dataset}-{self.excerpt}')
        fig.show()
# 


    '''
    Writes extracted onset events to output file 
    @param onset_times: list of [start, end] times
        out_file: file to write to 
    '''
    def write_output_files(self):
        print('--------Writing output files-------')
        with open(self.out_file, 'w', newline='\n') as out:
            # Write header
            fieldnames = ["OnSet", "Duration", "Notation"]
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()

            # Write each extracted onset event as a row
            for start_time, end_time in self.onset_times:
                writer.writerow({'OnSet': '%.3f' % start_time,
                                'Duration': '%.3f' % (end_time - start_time)})


        
        # initialize directories 
        init_dir(self.sequence_dir)

        # init positive dir
        if self.positive_dir:
            pos_dir = self.positive_dir 
            if not os.path.isdir(pos_dir): os.mkdir(pos_dir)
        else:
            pos_dir = self.sequence_dir + "positive/"
            init_dir(pos_dir)

        flatline_value = self.find_flatline_value("Orig_Value")
        self.df["ZScore"] = (self.df["Orig_Value"] - flatline_value) / self.df["Orig_Value"].std()
        self.df["ZScoreNorm"] = self.df["ZScore"].clip(lower=-5, upper=5)
        
        fig = self.visualize(value_field='ZScoreNorm')
        fig.update_layout(title=f'{self.dataset}-{self.excerpt}-ZScoreNorm')
   
        # apnea start time
        for start_time, end_time in self.onset_times:
            # File name is the start time 
            pos_out_file = f'{start_time}.txt'
            try:
                # slice from <SECONDS_BEFORE_APNEA> sec before apnea to <SECONDS_AFTER_APNEA> sec after
                start_idx = self.df.index[self.df["Time"] == round(start_time, 3)][0]
                end_idx =   self.df.index[self.df["Time"] == round(end_time, 3)][0]

                # Write to positive file VIVIVIVIVI
                self.df.iloc[start_idx:end_idx,  self.df.columns.get_loc('ZScoreNorm')].to_csv(pos_dir + pos_out_file,\
                                                    index=False, header=False, float_format='%.3f')
            except:
                continue


        
        # Initialize neg dir 
        if self.negative_dir:
            neg_dir = self.negative_dir
            if not os.path.isdir(neg_dir): os.mkdir(neg_dir)
        else:
            neg_dir = self.sequence_dir + "negative/"
            init_dir(neg_dir)
        
        # write negative sequences 
        for start_time, end_time in self.nononset_times: 
            
            # File name is start time 
            neg_out_file = f'{start_time}.txt' 

            try:
                # slice for <SECONDS_BEFORE_APNEA + SECONDS_AFTER_APNEA> seconds 
                start_idx = self.df.index[self.df["Time"] == round(start_time, 3)][0]
                end_idx   = self.df.index[self.df["Time"] == round(start_time + \
                            (self.seconds_before_apnea + self.seconds_after_apnea), 3)][0]

                # Write to negative file 
                self.df.iloc[start_idx:end_idx,  self.df.columns.get_loc('ZScoreNorm')].to_csv(neg_dir + neg_out_file,\
                                                index=False, header=False, float_format='%.3f')
            except:
                continue

 


''' Helper function to create directory '''
def init_dir(path): 
    if os.path.isdir(path): shutil.rmtree(path)
    if not os.path.isdir(path):
        os.makedirs(path)