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
import os

# ''' Visualize negatives'''
# sample_rate = 8
# path = "data/negative_pool/"
# files = os.listdir(path)
# start = 0
# while True:
#     batch_size = 4
#     figure, axes = plt.subplots(nrows=1, ncols=batch_size, figsize=(12,2), sharey=True)

#     for i in range(batch_size):
#         file = files[start + i]
#         arr = np.loadtxt(path+file, delimiter="\n", dtype=np.float64)
#         y = torch.from_numpy(zscore(arr))
#         n = len(arr)
#         x = torch.linspace(0, n/sample_rate, n)
#         axes[i].plot(x, y)
#         axes[i].set_title(f'{file}')
#     plt.show()
#     start += batch_size



sample_rate = 8
path = "data/dreams/postprocessing/excerpt7/positive/"
files = os.listdir(path)
start = 0
while True:
    batch_size = 4
    figure, axes = plt.subplots(nrows=1, ncols=batch_size, figsize=(12,2), sharey=True)

    for i in range(batch_size):
        file = files[start + i]
        arr = np.loadtxt(path+file, delimiter="\n", dtype=np.float64)
        y = torch.from_numpy(arr)
        n = len(arr)
        x = torch.linspace(0, n/sample_rate, n)
        axes[i].plot(x, y)
        axes[i].set_title(f'{file}')
    plt.show()
    start += batch_size