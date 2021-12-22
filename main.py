import numpy as np
import pandas as pd
import time
import random
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Loading dataset
#data = pd.read_csv('datset.csv')
scaler = StandardScaler()
filename = "New_DDOS_new.csv"
#n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
#s = 50000 #desired sample size
#skip = sorted(random.sample(range(1, n+2), n-s)) #the 0-indexed header will not be included in the skip list
df = pd.read_csv(filename)
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
df = df.drop(['Label_old'], axis=1)
df.fillna(df.mean(), inplace=True)
#print(df.head())
y = df["Label"]
X = df.drop(["Label"], axis=1)

# Splitting iris dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=13)
loadsize = X_test.memory_usage(index=True).sum()
#X_test = X_test[:8000]
#y_test = y_test[:8000]
print(X_test.shape)
print(X_train.shape)
X_train = X_train.to_numpy().reshape(X_train.shape)
X_test = X_test.to_numpy().reshape(X_test.shape)

#y_train = y_train.to_numpy().reshape(-1, 1)
#y_test = y_test.to_numpy().reshape(-1, 1)

X_train = scaler.fit_transform(X_train)
#y_train = scaler.fit_transform(y_train)

X_test = scaler.fit_transform(X_test)
#y_test = scaler.fit_transform(y_test)

# Creating model
#model = PassiveAggressiveClassifier(C=0.01, random_state=1)
#model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
model = GaussianNB()
print("Training Started")
# Fitting model
model.fit(X_train, y_train)

print("Prediction Started")
# Making prediction on test set
i=0
total_time = 0
while(i<=30):
    tick = time.time()
    test_pred = model.predict(X_test)
    total_time += time.time() - tick
    i +=1
print("Total Time: ",total_time/30)
print("Data size/Load: ", loadsize)
# Model evaluation
print(f"Test Set Accuracy : {accuracy_score(y_test, test_pred) * 100} %\n\n")

print(f"Classification Report : \n\n{classification_report(y_test, test_pred)}")