import pandas as pd 

from sklearn.neighbors import KNeighborsClassifier
import joblib

def train():
    train = pd.read_csv('classified_train.csv')
    X= []
    Y= []
    
    for i in range(train.shape[0]):
        x1 = list(train.iloc[i,:-1]) 
        y1 = train.iat[i,-1]
        X.append(x1)
        Y.append(y1)
        
        
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, Y)
    
    filename = 'trained_model.sav'
    joblib.dump(model, filename)

train()
    
    

    