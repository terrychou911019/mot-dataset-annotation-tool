import pandas as pd; 

df=pd.read_csv(f'gta_tracklets/seq01.txt',header=None); 

print(df[1].nunique())
