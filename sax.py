import pandas as pd
import numpy as np

# plot
import plotly
from plotly.offline import plot

import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import KBinsDiscretizer  


''' Apnea event annotation '''

import matplotlib.pyplot as plt
from scipy.stats import zscore


COLORS = ["red", "orange", "yellow", "green", "blue", "brown", "yellowgreen", "salmon", "aqua", "orchid"]

class SAX:
    def __init__(self, in_file, strategy="uniform", n_bins = 5, max_signal_len=100):
        self.in_file = in_file
        # signal
        self.signal = pd.read_csv(in_file, delimiter=',').iloc[:max_signal_len, :]
        self.signal_length = len(self.signal)
        self.values = np.expand_dims(self.signal["Value"].to_numpy(), -1)
        
        # parameters
        self.n_bins = n_bins
        self.strategy = strategy

        # map to chars
        self.chars = np.array([chr(i) for i in range(97, 97 + self.n_bins)])
        self.map_to_char = lambda i: self.chars[i]

    
    def transform(self):
        # normalize to mean = 0, stdev = 1
        self.values = zscore(self.values)
        # discretize into bins
        discretizer = KBinsDiscretizer(n_bins = self.n_bins,\
                                       strategy = self.strategy)
        transformed = discretizer.fit_transform(self.values)
        # map to chars
        indices = transformed.nonzero()[1]
        self.signal["Sax_Num"] = pd.Series(indices)
        self.chars = self.map_to_char(indices)
        self.signal["Sax_Char"] = pd.Series(self.chars)

    def visualize(self):
        # plot SAX
        self.signal.plot(x='Time', y="Sax_Num")
        for i, row in self.signal.iterrows():
            plt.text(row['Time'], row['Sax_Num'], str(row['Sax_Char']))
        plt.show()

    # def detect,

    def get_similarity(self):
        orig = self.chars[:10]
        print('orig', orig)
        i = 5
        k = 10 # window size 
        while (i + k) < self.signal_length: 
            seq = self.chars[i:i+k]
            seq_set = list(dict.fromkeys(seq))

            print(seq_set)
            i += 1



# text1 = "aabbbbdd"
# text2 = "abbbbdd"
# vector1 = text_to_vector(text1)
# vector2 = text_to_vector(text2)
# cosine = get_cosine(vector1, vector2)


in_file = "data/dreams/preprocessing/excerpt1/dreams_osa_ex1_sr8_sc1.txt"

s = SAX(in_file)
s.transform()
s.visualize()
s.get_similarity()
