import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier


#object creations with the hyperparameters i think which produces the best predictions
sgd_clf = SGDClassifier(random_state=42)
knn_clf = KNeighborsClassifier()
forest = RandomForestClassifier(random_state=42)
gbc = GradientBoostingClassifier(n_estimators =1000,learning_rate=0.01,max_depth=1,random_state=42)
xgb = XGBClassifier(n_estimators =1000,learning_rate=0.01,max_depth =3,random_state=42)


#Getting the data
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


#Model :Voting Classifier
from sklearn.ensemble import VotingClassifier
voting_clf1 = VotingClassifier(estimators=[('SGD',sgd_clf,),('kNN',knn_clf),('Forest',forest),('GBC',gbc),('XGB',xgb)],voting='hard')
voting_clf1.fit(df_train_data,df_train_label)
print("Prediciting #1")
hard_p = voting_clf1.predict(df_test)

df_id = pd.read_csv("data/test.csv")
submission1= pd.DataFrame({
    "PassengerId": df_id["PassengerId"],
    "Survived": hard_p.astype(int)
})

submission1.to_csv("results_hard.csv", index = False)


