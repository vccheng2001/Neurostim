import os, sys
import numpy as np
from numpy import mean, std, dstack 
import pandas as pd

from matplotlib import pyplot as plt

(program, apnea_type) = sys.argv
def graph():
    pred_path = "predictions_window/" # predicted flags on 3rd col 
    actual_path = f"test_{apnea_type}/" # actual flags
    images_path = "images5/"
    make_dir(images_path)
    for i in range(1,10):
        try:
            file_name = f"e{i}_osa"
            pred_file = f"{pred_path}{file_name}.txt" 
            actual_file = f"{actual_path}{file_name}.txt" 
            df_pred = pd.read_csv(pred_file, delim_whitespace=True, usecols=[2])
            df_actual = pd.read_csv(actual_file, delim_whitespace=True,usecols=[1])
            
            plt.plot(df_pred[10000:15000:10], color="blue",label="pred")
            plt.plot(df_actual[10000:15000:10], color="orange",label="actual")
            plt.title(file_name)
            # filler = [0]*(int(timesteps))
            # filler.extend(flags)
            # plt.plot(flags[2000:8000:2],color="blue",label="predictions")
            # plt.plot(actual[2000:8000:2],color="orange",label="actual")
            plt.legend()
            # plt.show()
            plt.savefig(f"{images_path}{file_name}.png")
        except Exception as e:
            print(e)
            continue 

# Makes directory 
def make_dir(path):
    if not os.path.isdir(path):
        print("Making dir.... " + path)
        os.mkdir(path)
        
if __name__ == "__main__":
    graph()