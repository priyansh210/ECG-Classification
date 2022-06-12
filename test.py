import pandas as pd 
import numpy as np 
import heartpy as hp
import joblib




def classify():
    df = pd.read_csv("testing_set.csv")

    train = pd.DataFrame(columns = ['bpm', 'ibi', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad','class'])
    length = df.shape[0]
    a=0
    for r in range(length):
        c = 0 if df.iat[a,1]=='N' else 1

        y = np.array(list(map(int,df.iloc[a,0].split(','))))
        try:
            working_data, features = hp.process(y, sample_rate=300.0, calc_freq=False)
            features = { k:features[k] for k in list(features.keys())[:8]}
            features['class']  = c
            features = [0 if (type(i)==str or i!=i or np.ma.is_masked(i) or i=='') else i for i in list(features.values())]
            train.loc[a] = features
            a+=1
    
    
        except:
            pass
    
    train.to_csv('classified_test.csv',index=False)
    
# classify()


def predict():
    model = joblib.load('trained_model.sav') #importing the model
    test = pd.read_csv('classified_test.csv')
    length = test.shape[0]
    checkdf = pd.DataFrame(columns = ['Predict' , 'Ans' ])
    row=0
    for i in range(length):
        
    
        x = list(test.iloc[i,:-1]) 
        y = test.iat[i,-1]
      
        # print(neigh.predict_proba([ftest]))
        checkdf.loc[row] = [model.predict([x])[0] , y]
        row+=1
    
        
    checkdf.to_csv('check.csv',index=False)
    
# predict()
    
def evaluate():
    
    main = pd.read_csv('check.csv')
    ans = list(main['Ans'])
    test = list(main['Predict'])
    
    
            
    f = pd.DataFrame({'TF':ans,'PN':test})
    TP= len(f[(f['TF']==0) & (f['PN']==0)])
    FN= len(f[(f['TF']==0) & (f['PN']==1)])
    FP= len(f[(f['TF']==1) & (f['PN']==0)])
    TN= len(f[(f['TF']==1) & (f['PN']==1)])
    
    
    
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precission = TP/(TP+FP)
    recall = TP/(TP+FN)
    
    F1 = 2*precission*recall/(precission+recall)
    matrix = (np.array([[TP,FN],[FP,TN]]),'\n')
    op = {'acc':accuracy,'matrix':matrix,'f1':F1}
    return op

t=evaluate()
print(t)