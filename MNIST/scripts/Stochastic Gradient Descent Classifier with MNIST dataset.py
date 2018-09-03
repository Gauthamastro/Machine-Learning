import pickle
import pandas as pd
from sklearn.linear_model import SGDClassifier

#loading the preproccessed data from pickle
label = pickle.load(open("augmented.labels","rb"))
target = pickle.load(open("augmented.data","rb"))
data_test = pd.read_csv("test.csv")

#initializing and training the Stochastic Gradient Descent Classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(target,label)
print("Trained")
a = sgd_clf.predict(data_test)

#Writing the predictions to a file as per the format of kaggle competitions
with open('results_gj.csv', 'w') as the_file:
    the_file.write('ImageId,Label\n')
    for i in range(28000):
        j = str(i+1)+','+str(a[i])+'\n'
        the_file.write(j)





