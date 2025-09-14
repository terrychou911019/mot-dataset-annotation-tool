import pandas as pd; 

tracklet_id = 18

df=pd.read_csv(f'gta_tracklets/seq01.txt',header=None); 

# show how many frames this tracklet id appears in
print(df[df[1]==tracklet_id].shape[0])

