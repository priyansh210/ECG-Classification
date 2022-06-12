import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

df = pd.read_csv('ECG_training.csv')
i=4     #total graphs max 12 graphs
n=2   #graphs per row
fig , axs = plt.subplots(i//n , n)


if i>12:
    i=12
    
    
for a in range(i):
    y = np.array(list(map(int,df.iloc[a,0].split(','))))
    x = np.linspace(0,8999,9000)
    
    if df.iat[a,1]=='N':
        t = 'NORMAL'
        c = 'blue'
        
    else :
        t = "AF Episode"
        c = 'red'
        
    
    axs[a//n, a%n].plot(x, y,color = c)
    axs[a//n, a%n].set_title(t)
    
        

    
plt.show()