import pandas as pd

def preprocessing(x):
    ti = pd.read_csv(x)
    #Removing Non needed data
    drop_col = ['Cabin','PassengerId','Name','Ticket']
    ti.drop(drop_col, axis=1, inplace=True)
    #Converting text to number for better perf in training
    ti.loc[ti['Sex'] == 'male', 'Sex'] = 1
    ti.loc[ti['Sex'] == 'female', 'Sex'] = 0
    ti.loc[ti['Embarked'] == 'S', 'Embarked'] = 0
    ti.loc[ti['Embarked'] == 'Q', 'Embarked'] = 1
    ti.loc[ti['Embarked'] == 'C', 'Embarked'] = 2
    #OneHotEncoding
    one_hot = pd.get_dummies(ti['Embarked'])
    ti.drop('Embarked',axis =1 ,inplace=True)
    ti= ti.join(one_hot)
    #Data Cleaning
    from sklearn.preprocessing import Imputer
    imputer = Imputer(strategy ="median")
    imputer.fit(ti)
    x = imputer.transform(ti)
    ti_trans = pd.DataFrame(x,columns= ti.columns)
    return ti_trans 

#Prepping data
df_train = preprocessing("data/train.csv")
df_test = preprocessing("data/test.csv")

#Some more preprocessing...
df_train_label = df_train["label"]
df_train.drop("label",axis =1,inplace=True)
df_train_data = df_train
print(df_train_data.head())
print(df_train_label.head())
#Trying different models
    
def SGDC(df_train_data,df_train_label,df_test):  #Stochastic Gradient Descent Classifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import cross_val_score
    sgd_clf = SGDClassifier(random_state=42)
    print("Training...")
    sgd_clf.fit(df_train_data,df_train_label)
    print("Predicting...")
    return sgd_clf.predict(df_test),cross_val_score(sgd_clf,df_train_data,df_train_label,cv=5,scoring='accuracy')

def Knearest(df_train_data,df_train_label,df_test):  #K-Nearest Neighbor Classifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    knn_clf = KNeighborsClassifier()
    print("Training...")
    knn_clf.fit(df_train_data,df_train_label)
    print("Prediciting...")
    return knn_clf.predict(df_test),cross_val_score(knn_clf,df_train_data,df_train_label,cv=5,scoring='accuracy')

def RandomForest(df_train_data,df_train_label,df_test): #Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    forest = RandomForestClassifier(random_state=42)
    print("Training...")
    forest.fit(df_train_data,df_train_label)
    print("Prediciting...")
    return forest.predict(df_test),cross_val_score(forest,df_train_data,df_train_label,cv=5,scoring='accuracy')
    

def GradientBoostingClassifier(df_train_data,df_test_label,df_test):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    gbc = GradientBoostingClassifier(n_estimators =100,learning_rate=0.01,max_depth=1,random_state=42)
    gbc.fit(df_train_data,df_test_label)
    return gbc.predict(df_test),cross_val_score(gbc,df_train_data,df_test_label,cv=5,scoring='accuracy')


def XGBoost(df_train_data,df_test_label,df_test):
    from xgboost.sklearn import XGBClassifier
    from sklearn.model_selection import cross_val_score
    xgb = XGBClassifier(n_estimators =1000,learning_rate=0.01,max_depth=3,random_state=42)
    xgb.fit(df_train_data,df_test_label)
    return xgb.predict(df_test),cross_val_score(xgb,df_train_data,df_test_label,cv=5,scoring='accuracy')

#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import GradientBoostingClassifier
#gbct = GradientBoostingClassifier()
#n_estimator =[100,200,500,1000]
#learning_rates =[1,0.1,0.01]
#max_depths=[1,3,5]
#grid = GridSearchCV(estimator=gbct,param_grid=dict(n_estimators=n_estimator,learning_rate=learning_rates,max_depth=max_depths))
#grid.fit(df_train_data,df_train_label)
#print(grid)
#print(grid.best_score_)
#print(grid.best_estimator_.n_estimators)
#print(grid.best_estimator_.learning_rate)
#print(grid.best_estimator_.max_depth)

print("Test #1  - SGDC")
sgdc,score1 = SGDC(df_train_data,df_train_label,df_test)
print("Test #2  - Knearest")
knn,score2 = Knearest(df_train_data,df_train_label,df_test)
print("Test #3  - RandomForest")
rf,score3 = RandomForest(df_train_data,df_train_label,df_test)
print("Test #4 - GradientBoostingClassifier")
gbc,score4 = GradientBoostingClassifier(df_train_data,df_train_label,df_test)
print("Test #5  - XGBoostClassifier")
xgb,score5 = XGBoost(df_train_data,df_train_label,df_test)

def mean(x):
    k = 0
    for i in x:
        k = k + i
    return (k/5)

print("SGDC : ",score1)
print("Mean accuracy : " ,mean(score1))
print("KNN : ",score2)
print("Mean accuracy : " ,mean(score2))
print("Forest : ",score3)
print("Mean accuracy : " ,mean(score3))
print("GradientBoosting :",score4)
print("Mean accuracy : " ,mean(score4))
print("XGBoost: ",score5)
print("Mean accuracy : " ,mean(score5))


#Code for generating the kaggle-formatted file with predictions for their test dataset.

df_id = read_data("data/test.csv")
#Submission #1
submission1= pd.DataFrame({
    "PassengerId": df_id["PassengerId"],
    "Survived": rf.astype(int)
})

submission1.to_csv("results_forest.csv", index = False)

#Submission #2
submission2 = pd.DataFrame({
    "PassengerId": df_id["PassengerId"],
    "Survived": sgdc.astype(int)
})

submission2.to_csv("results_sgdc_gj.csv", index = False)

#Submission #3
submission3 = pd.DataFrame({
    "PassengerId": df_id["PassengerId"],
    "Survived": knn.astype(int)
})

submission3.to_csv("results_knn_gj.csv", index = False)

#Submission #4
submission4= pd.DataFrame({
    "PassengerId": df_id["PassengerId"],
    "Survived": gbc.astype(int)
})

submission4.to_csv("results_GBC.csv", index = False)

#Submission #5
submission5= pd.DataFrame({
    "PassengerId": df_id["PassengerId"],
    "Survived": xgb.astype(int)
})

submission5.to_csv("results_XGB.csv", index = False)



