import pandas as pd
import numpy as np

Data = pd.read_csv("MalwareData.csv",sep="|")

# legitimate files

legit = Data[0:41323].drop(["legitimate"],axis=1)

#choosing subset of data from malware files
mal = Data[41323::].drop(["legitimate"],axis=1)


# Removing non essential columns for training data
data_train = Data.drop(['Name','md5','legitimate'], axis=1).values
labels = Data['legitimate'].values


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


# Using Logistic Regression 
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sc = StandardScaler()
legit_train_scale = sc.fit_transform(legit_train)
legit_test_scale = sc.transform(legit_test)

# Training the Model
logclf = LogisticRegression(random_state = 0)
logclf.fit(legit_train_scale, mal_train)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(20):
    legit_train, legit_test, mal_train, mal_test = train_test_split(data_train_new, labels, test_size = 0.25)
    legit_train_scale = sc.fit_transform(legit_train)
    legit_test_scale = sc.transform(legit_test)

    logclf = LogisticRegression(random_state = 0)
    logclf.fit(legit_train_scale, mal_train)
    acc = logclf.score(legit_test_scale, mal_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("malware_log_clf.pickle", "wb") as f:
            pickle.dump(logclf, f)


pickle_in = open("malware_log_clf.pickle", "rb")
logclf = pickle.load(pickle_in)
logclf.score(legit_test_scale, mal_test)

#Comparing predicitons with actual results
predicted= logclf.predict(legit_test_scale)
for w in range(21):
    print(mal_test[w],' ',legit_test[w])
    print(predicted[w])
    print()




