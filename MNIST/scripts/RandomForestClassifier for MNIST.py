#MNIST using RandomForest classifier
import pickle
import pandas as pd
x_train = pickle.load(open("augmented.data","rb"))
y_train = pickle.load(open("augmented.labels","rb"))
data_test = pd.read_csv("test.csv")

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
print("Training")
forest_clf.fit(x_train,y_train)
print("Trained")
a = forest_clf.predict(data_test)

#Writing predictions to a file for kaggle upload
with open('results_gj_Randomforest.csv', 'w') as the_file:
    the_file.write('ImageId,Label\n')
    for i in range(28000):
        j = str(i+1)+','+str(a[i])+'\n'
        the_file.write(j)
print(a)    
print("Finished")
