import csv
import os
import pandas as pd
from pathlib import Path
import shutil

ROOT_DIR = '.'
DATA_DIR = os.path.join(ROOT_DIR, "data")


def init_dir(path): 
    if os.path.isdir(path): shutil.rmtree(path)
    if not os.path.isdir(path):
        os.mkdir(path)

def combine_data(dataset, apnea_type, excerpts, sample_rate, scale_factor):

    combined_excerpt = ''.join(excerpts)
    prev_excerpt = excerpts.pop(0)
    prev_file = f"{DATA_DIR}/{dataset}/preprocessing/excerpt{prev_excerpt}/{dataset}_{apnea_type}_ex{prev_excerpt}_sr{sample_rate}_sc{scale_factor}.csv"
 
    prev_df = pd.read_csv(prev_file, delimiter=',')
    while excerpts:
        prev_time = prev_df["Time"].iloc[-1] 
        curr_excerpt = excerpts.pop(0)
        curr_file = f"{DATA_DIR}/{dataset}/preprocessing/excerpt{curr_excerpt}/{dataset}_{apnea_type}_ex{curr_excerpt}_sr{sample_rate}_sc{scale_factor}.csv"
        curr_df = pd.read_csv(curr_file, delimiter=',')
        curr_df["Time"] += prev_time
        prev_df = prev_df.append(curr_df, ignore_index=True)

    init_dir(f"{DATA_DIR}/{dataset}/preprocessing/excerpt{combined_excerpt}")

    new_csv= f"{DATA_DIR}/{dataset}/preprocessing/excerpt{combined_excerpt}/{dataset}_{apnea_type}_ex{combined_excerpt}_sr{sample_rate}_sc{scale_factor}.csv"
    prev_df.to_csv(new_csv, sep=',', index=None)

combine_data('patch', 'osa', ['1','2','3'], 8, 1)