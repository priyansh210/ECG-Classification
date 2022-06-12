import pandas as pd 
import numpy as np 
import heartpy as hp


df = pd.read_csv("training_set.csv")

length = df.shape[0]
train = pd.DataFrame(columns = ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad','class'])

for a in range(length):
    c = 0 if df.iat[a,1]=='N' else 1
    y = np.array(list(map(int,df.iloc[a,0].split(','))))
    try:
        working_data, features = hp.process(y, sample_rate=300.0, calc_freq=False)
        features = { k:features[k] for k in list(features.keys())[:8]}
        features['class']  = c
        features = [0 if (type(i)==str or i!=i or np.ma.is_masked(i) or i=='') else i for i in list(features.values())]
        train.loc[a] = features
      
    except:
        pass

train.to_csv('classified_train.csv',index=False)