import pandas as pd
file = "data/patch/preprocessing/excerpt1/filtered_16hz.txt"
df = pd.read_csv(file)
del_column = df.columns[1]
df = df.drop([del_column], axis=1)

df.to_csv(file, index=None)