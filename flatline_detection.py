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


# read unnormalized file
file = f"{DATA_DIR}/{dataset}/preprocessing/excerpt{excerpt}/filtered_{sample_rate}hz.txt"
df = pd.read_csv(file, delimiter=',')
#plot
df = df.iloc[2000:7000]
df['Time1'] = df['Time'] - 200
df.plot(x ='Time1', y='Value', kind = 'line')
plt.show()

# difference of values 10 sec apart 
df['Diff'] = df['Value'].diff(10)
# set to 0 if < THRESHOLD, else 1
df['Diff'] = np.where(abs(df['Diff']) >= FLATLINE_THRESHOLD, 1, 0)
# convert to binary string representation
bin_list = df['Diff'].tolist()
bin_str = ''.join(str(int(x)) for x in bin_list)

# only mark as flatline if continuous flatline for 10 (10 seconds) 
flatlines = [(m.start()/10, m.end()/10) for m in re.finditer(r"(0)\1{100,}", bin_str)]
print(flatlines)
