
import os
import pandas as pd
import numpy as np
import csv
from dataloader import ApneaDataloader

from cnn import CNN
import torch
from torch import nn

import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from scipy.stats import zscore
from scipy import fftpack
np.set_printoptions(suppress=True) # don't use scientific notation


def rolling_window(x, window_size, step_size=1):
    # unfold dimension to make our rolling window
    return x.unfold(0,window_size,step_size)

class StreamPrediction():
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
        # default parameters 
        self.seconds_before_apnea = 10
        self.seconds_after_apnea = 5

        self.timesteps = self.sample_rate * (self.seconds_before_apnea + self.seconds_after_apnea)

    def setup(self, model, save_model_path):
        # Set up sequence 
        self.df = pd.read_csv(self.in_file, delimiter=",")
        # self.df = self.df.iloc[:300]
        # self.df["Value"] = zscore(self.df["Value"])
        self.seq = torch.from_numpy(self.df["Value"].to_numpy())
        self.time = torch.from_numpy(self.df["Time"].to_numpy())

        # slide offset: 1 sec
        self.values = rolling_window(self.seq, self.timesteps*2, self.sample_rate*10).unsqueeze(-1)
        self.times = rolling_window(self.time, self.timesteps*2, self.sample_rate*10).unsqueeze(-1)


        self.seq_len = len(self.df)
        self.start = 0

        # Model
        self.model = model.double()
        # self.model.load_state_dict(torch.load(save_model_path))
        
    def do_fft(self):
        while True:
            fig, axs = plt.subplots(2)
            data = self.values[self.start].numpy()
            axs[0].plot(self.times[self.start], self.values[self.start])
            n = float(len(data))
            T = float(1/8)
            yf = fftpack.fft(data)
            xf = np.linspace(0, 1.0/(2.0*T), num=int(n/2))
            fig, ax = plt.subplots()
            axs[1].plot(xf, 2.0/n * np.abs(yf[:int(n/2)]))
            plt.grid()
            plt.xlabel("Freq")
            plt.ylabel("Mag")
            plt.show()
            self.start += 16

    # def fft_plot(self,audio, sample_rate):
    #     n = float(len(audio))
    #     T = float(1/sample_rate)
    #     yf = fftpack.fft(audio)
    #     xf = np.linspace(0, 1.0/(2.0*T), num=int(n/2))
    #     fig, ax = plt.subplots()
    #     ax.plot(xf, 2.0/n * np.abs(yf[:int(n/2)]))
    #     plt.grid()
    #     plt.xlabel("Freq")
    #     plt.ylabel("Mag")
    #     return plt.show()

    def predict_next(self):
        
        batch_size = 16
        np.random.seed(1234)

        time_step = 0.02
        period = 5.

        time_vec = np.arange(0, 20, time_step)
        sig = (np.sin(2 * np.pi / period * time_vec)
            + 0.5 * np.random.randn(time_vec.size))

        plt.figure(figsize=(6, 5))
        plt.plot(time_vec, sig, label='Original signal')

        # The FFT of the signal
        sig_fft = fftpack.fft(sig)

        # And the power (sig_fft is of complex dtype)
        power = np.abs(sig_fft)**2

        # The corresponding frequencies
        sample_freq = fftpack.fftfreq(sig.size, d=time_step)

        # Plot the FFT power
        plt.figure(figsize=(6, 5))
        plt.plot(sample_freq, power)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('plower')

        # Find the peak frequency: we can focus on only the positive frequencies
        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        peak_freq = freqs[power[pos_mask].argmax()]

        # Check that it does indeed correspond to the frequency that we generate
        # the signal with
        np.allclose(peak_freq, 1./period)

        # An inner plot to show the peak frequency
        axes = plt.axes([0.55, 0.3, 0.3, 0.5])
        plt.title('Peak frequency')
        plt.plot(freqs[:8], power[:8])
        plt.setp(axes, yticks=[])

        high_freq_fft = sig_fft.copy()
        high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
        filtered_sig = fftpack.ifft(high_freq_fft)

        plt.figure(figsize=(6, 5))
        plt.plot(time_vec, sig, label='Original signal')
        plt.plot(time_vec, filtered_sig, linewidth=3, label='Filtered signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')

        plt.legend(loc='best')
        ''''''

        # scipy.signal.find_peaks_cwt can also be used for more advanced
        # peak detection




        # figure, axes = plt.subplots(nrows=1, ncols=4, figsize=(12,2), sharey=True)

        # batch windows, slide vertically 
        from scipy.fft import fft,fftfreq
        while True:
            inp =  self.values[self.start]

            fig, axs = plt.subplots(2)
            fig.suptitle('Vertically stacked subplots')
            axs[0].plot(self.times[self.start], self.values[self.start])


            yf = fft(inp.numpy())
            xf = fftfreq(120, 1/8)  
            axs[1].plot(xf, np.abs(yf))
            plt.show()
            self.start += batch_size

        # inp =  self.values[self.start:self.start + batch_size, :, :]
        # pred = self.model(inp)
        # pred_bin = pred_bin = torch.argmax(pred, dim=1)
        
        # # plot 
        # for i in range(batch_size):
        #     t = self.times[self.start+i, :, :]
        #     v = self.values[self.start+i, :, :]
        #     axes[i].plot(t.squeeze().numpy(), v.squeeze().numpy())
        #     axes[i].set_title(f'{pred_bin[i]}')
        # plt.show()
        # self.start += 4
        # return pred_bin

        # curr_window = torch.Tensor(curr_window.to_numpy())
        # curr_window = curr_window.unsqueeze(0).unsqueeze(-1).double()

        # print(curr_window)
        # print(curr_window.shape)
        # pred, label = self.model(curr_window)
        # window_offset = self.sample_rate * 1 # Sample rate * num seconds
        # self.start += window_offset # shift one second 

        # # pd.options.plotting.backend = "plotly"
        # title = f"Pred: {pred}, label: {label}"
        # fig = curr_window.plot(x='Time', y="Value", title=title)
        # fig.show()

        # return pred, label


def main(args):
    sp = StreamPrediction(root_dir=".",
                        dataset=args.dataset,
                        apnea_type=args.apnea_type,
                        excerpt= args.excerpt,
                        sample_rate=args.sample_rate,
                        scale_factor=args.scale_factor)
    
    model =  CNN(input_size=1, \
                output_size=2).double()
    sp.setup(model=model, save_model_path='base_model.ckpt')
    
    sp.do_fft()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir",    default=".", help="root directory (parent folder of data/)")
    parser.add_argument("-d", "--dataset",    default="dreams", help="dataset (dreams, dublin, or mit)")
    parser.add_argument("-a", "--apnea_type", default="osa",    help="type of apnea (osa, osahs, or all)")
    parser.add_argument("-ex","--excerpt",    default=1,        help="excerpt number to use")
    parser.add_argument("-sr","--sample_rate",    default=8,        help="number of samples per second")
    parser.add_argument("-sc","--scale_factor",    default=1,        help="scale factor for normalization")

    # parse args 
    args = parser.parse_args()
    main(args)