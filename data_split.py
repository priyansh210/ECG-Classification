import pandas as pd 

main = pd.read_csv('ECG_training.csv')
training_set = main.iloc[:3000]
testing_set = main.iloc[3000:]

testing_set.to_csv('testing_set.csv',index=False)
training_set.to_csv('training_set.csv',index=False)


# test = pd.read_csv('testing_set.csv')
# print(test.head(5))
# # test.reset_index(inplace=True)
# # test.to_csv('testing_set.csv')