from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
import csv
import datetime as dt
from cnn import CNN
from apnea_detection import DefaultConfig
import torch
from scipy.stats import zscore

from scipy.fft import fft, fftfreq

'''
No pos/neg sequences needed
Real time
Robust to noise
only 1 parameter
prone to false positives
just one script

'''


# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

cfg = DefaultConfig(dataset = "dreams",
                    apnea_type = "osa",
                    excerpt = "6",
                    sample_rate = 8,
                    root_dir = ".",
                    seconds_before_apnea = 10,
                    seconds_after_apnea = 5,
                    base_model_path="base_model.ckpt")

base_path = f"{cfg.root_dir}/data/{cfg.dataset}/preprocessing/excerpt{cfg.excerpt}/{cfg.dataset}_{cfg.apnea_type}_ex{cfg.excerpt}_sr{cfg.sample_rate}"
in_file     = f"{base_path}_sc1.csv"
df = pd.read_csv(in_file, delimiter=",")
df = df.iloc[30000:40000]

model = CNN(input_size=1,output_size=2)
model = model.double()
model.load_state_dict(torch.load("dreams_model.ckpt"))
model.eval()


out_file = f"testout_{cfg.dataset}_{cfg.apnea_type}_ex{cfg.excerpt}_sr{cfg.sample_rate}.txt"
out = open(out_file, 'a', newline='\n')
i = 0
def animate(i, xs, ys):
    i += 1


    # Add x and y to lists
    time = df['Time'].iloc[i]
    val = df['Value'].iloc[i]
    xs.append(time)
    ys.append(val)




    # Limit x and y list size
    max_size = cfg.sample_rate * (cfg.seconds_before_apnea + cfg.seconds_after_apnea) # 80
    max_size_double = max_size * 10# 800



    # most recent 15 
    xs = xs[-max_size_double:]
    ys = ys[-max_size_double:]

 
    window = ys[-max_size:]
   


    if (i+1) %max_size == 0:

        # window = zscore(window)

        # array, 
        # b, t, c
        # inp = np.loadtxt("data/dreams/postprocessing/excerpt4/positive/7783.375.txt",delimiter="\n", dtype=np.float64)
        # inp = np.loadtxt("data/negative_pool/226.375.txt",delimiter="\n", dtype=np.float64)
        inp = torch.DoubleTensor(window)
        # inp = torch.DoubleTensor(inp)
        inp = inp.unsqueeze(0).unsqueeze(-1).repeat(2,1,1)
        
        # print('inp', inp.shape, inp)
        pred = model(inp)
        pred_bin = torch.argmax(pred, dim=1)[0]
        print(f'#{i}, pred: {pred}, pred_bin: {pred_bin}')
        if pred_bin == 1:
            print('********ALERT*******')
            ax.clear()
            ax.plot(xs, ys, 'r')
        else:
            ax.clear()
            ax.plot(xs, ys, 'g')

        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.25)
        plt.title('Apnea Detection')
        plt.xlabel(f'Timestamp (seconds)')
        plt.ylabel('Signal Value')


    '''
    # Limit x and y list size
    max_size = cfg.sample_rate * (cfg.seconds_before_apnea + cfg.seconds_after_apnea)
    max_size_double = max_size * 10

    # most recent 15 
    xs = xs[-max_size_double:]
    ys = ys[-max_size_double:]

    # variance (oldest 15)
    last15 = ys[-max_size:]
    # oldest 10 seconds
    var_before  = np.var(last15[:cfg.sample_rate * cfg.seconds_before_apnea])
    # most recent 5 seconds
    var_after = np.var(last15[-cfg.sample_rate * cfg.seconds_after_apnea:])
    print('Var before', var_before, 'Var after', var_after)


    if var_before > var_after * 10:
        print('****** ALERT ************ {i}')

        onset_times.append(time)
        out.write(f'{time}\n')
        # Draw x and y lists
        ax.clear()
        ax.plot(xs, ys, 'r')
    else:
        ax.clear()
        ax.plot(xs, ys, 'g')

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.25)
    plt.title('Apnea Detection')
    plt.xlabel(f'Timestamp (seconds)')
    plt.ylabel('Signal Value')

    '''

onset_times = []
# Set up plot to call animate() function periodically
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1)
plt.show()
