#getting the data
import pandas as pd
import pickle
import numpy
#from sklearn.linear_model import SGDClassifier

data = pd.read_csv("train.csv")
data = data.reset_index().values
data_new = []
for i in range(0,42000,1):
    a = numpy.delete(data[i],[0,1])
    data_new.append(a)
    print(i)
print("looped")
print("pickling")
pickle.dump(data_new, open("target.matrixlist","wb"))
