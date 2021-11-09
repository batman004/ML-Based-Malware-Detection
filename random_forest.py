
import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

Data = pd.read_csv("MalwareData.csv",sep="|")

# legitimate files

legit = Data[0:41323].drop(["legitimate"],axis=1)

#choosing subset of data from malware files
mal = Data[41323::].drop(["legitimate"],axis=1)


print("shape of legit dataset :",legit.shape[0],"samples",legit.shape[1],"features")
print("shape of malware dataset :",mal.shape[0],"samples",mal.shape[1],"features")

# 56 features to define the to define whether the sample is legit or a malware (excluding "legitimate")
print(Data.columns)

Data.head(10)

legit.head(10)


# Removing non essential columns for training data
data_train = Data.drop(['Name','md5','legitimate'], axis=1).values
labels = Data['legitimate'].values

# using ExtraTrees classifier to get most important features for training
extratrees = ExtraTreesClassifier().fit(data_train,labels)
select = SelectFromModel(extratrees, prefit=True)
data_train_new = select.transform(data_train)
print(data_train.shape, data_train_new.shape)

# number of selected features for training 
features = data_train_new.shape[1]
importances = extratrees.feature_importances_
indices = np.argsort(importances)[::-1]

# sorting the features according to its importance (influence on final result)
for i in range(features):
    print("%d"%(i+1), Data.columns[2+indices[i]],importances[indices[i]])


legit_train, legit_test, mal_train, mal_test = train_test_split(data_train_new, labels, test_size = 0.25)

#initialising a RandomForestClassifier model with 50 trees in the forest

randomf =RandomForestClassifier(n_estimators=50)

# training the model

randomf.fit(legit_train, mal_train)

# checking performance of the model 

print("Score of algo :", randomf.score(legit_test, mal_test)*100)

from sklearn.metrics import confusion_matrix

result = randomf.predict(legit_test)

''''The first number of the first matrix gives the number of correct predictions of that 
particular result which should be obtained and the second number gives the number of incorrect predictions made. 
Similarly vice versa for the second matrix present'''

conf_mat = confusion_matrix(mal_test,result)
print(conf_mat)







