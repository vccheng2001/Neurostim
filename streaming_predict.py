import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
import datetime as dt
from cnn import CNN
from apnea_detection import DefaultConfig
import torch
from scipy.stats import zscore


''' 
This program performs real-time apnea detection on input signal stream. 
The user must:
    (1) define a config (dataset, excerpt, ....)
    (2) define an input file to read input stream
    (3) define a trained model to make predictions (must specify a path to the model .ckpt file)
 '''


# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# store list of timestamps, values
xs = []
ys = []

''' (1) --------------------------------- config ------------------------------'''
cfg = DefaultConfig(logger=None, dataset = "patchAllisterW",
                    apnea_type = "osa",
                    excerpt = "14",
                    sample_rate = 8,
                    root_dir = ".",
                    seconds_before_apnea = 10,
                    seconds_after_apnea = 5,
                    base_model_path="patch_model_11.ckpt")

''' (2) ---------------------load input file to read -----------------------------'''
base_path = f"{cfg.root_dir}/data/{cfg.dataset}/preprocessing/excerpt{cfg.excerpt}/{cfg.dataset}_{cfg.apnea_type}_ex{cfg.excerpt}_sr{cfg.sample_rate}"
in_file   = f"{base_path}_sc1.csv"
df = pd.read_csv(in_file, delimiter=",")


''' (3) ---------------------define model, load trained parameters -----------------'''
model = CNN(input_size=1,output_size=2)
model = model.double()
model.load_state_dict(torch.load(cfg.base_model_path))
model.eval()

# output file to write detected onset times 
out_file = f"out_{cfg.dataset}_{cfg.apnea_type}_ex{cfg.excerpt}_sr{cfg.sample_rate}.txt"
out = open(out_file, 'a', newline='\n')
# number of timesteps to feed into model input (default: 15 seconds)
max_size = cfg.sample_rate * (cfg.seconds_before_apnea + cfg.seconds_after_apnea)
# length of entire graph x-axis (user-defined) 
max_size_double = max_size * 20
# indexer 
i = 0
# number of new data points to append to the current stream each time animate() is called
batch_size = 16
def animate(i, xs, ys):
    i += batch_size

    # add next <batch_size> timestamps, values to list 
    time = df['Time'].iloc[batch_size*i:batch_size*(i+1)].tolist()
    val = df['Value'].iloc[batch_size*i:batch_size*(i+1)].tolist()

    xs.extend(time)
    ys.extend(val)

    # normalize window
    if i > max_size:
        ys = zscore(ys)

    # length of entire graph x-axis (user-defined)
    xs = xs[-max_size_double:]
    ys = ys[-max_size_double:]

    # model input is most recent 15 seconds 
    window = ys[-max_size:]

    # wait for list to populate 
    if i > max_size:
        # input to model is window of 15 seconds, dimensions (b, t, c)
        inp = torch.DoubleTensor(window)
        # note: repeat is just to satisfy model's batchnorm req. of batchsize > 1
        inp = inp.unsqueeze(0).unsqueeze(-1).repeat(2,1,1)
        # model prediction 
        pred = model(inp)
        # print(f'Onset probability: {pred[0,1]}')


        # if detected apnea event 
        if pred[0,1] > 0.99:
            # print('******** detected apnea event *******')
            ax.clear()
            ax.plot(xs, ys, 'r')
        else:
            ax.clear()
            ax.plot(xs, ys, 'g')

        # plot 
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.25)
        plt.title('Apnea Detection')
        plt.xlabel(f'Timestamp (seconds)')
        plt.ylabel('Signal Value')
        plt.ylim([-25, 25])


# list to store detected onset times 
onset_times = []
# set up plot to call animate() function periodically
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys),interval=1)
plt.show()
