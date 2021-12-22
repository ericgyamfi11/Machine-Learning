import sys

from twoClassSVDD import TwoClassSVDD

sys.path.append("..")
from src.svdd import SVDD
from src.visualize import Visualization as draw
from data import PrepareData as load
import pandas as pd
from Utils.utils import EvalC as ev
from Utils.utils import EvalR as er
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.datasets import make_regression
from datetime import datetime
import pickle
from sklearn import metrics


#data = pd.read_csv('UNSW_NB1.csv')
#y = data['label']
#X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.7, random_state = 42)
#data = X_train
#y = data['label']
#data = data.drop(['label'], axis=1)
#X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.35, random_state = 42)
#plt.scatter(df[:, 0], df[:, 1])
#plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'coolwarm', s = 2)
#X_train = (X_train).to_numpy()
#y_train = (y_train).to_numpy()
#tcl_dsvdd = TwoClassSVDD(kernel='rbf').fit(X_train, y_train)
#pred = tcl_dsvdd.predict(X_test)

trainData, testData, trainLabel, testLabel = load.unsw()

# set SVDD parameters
parameters = {"positive penalty": 0.9, "negative penalty": 0.8,"kernel": {"type": 'lapl', "width": 1/24}, "option": {"display": 'on'}}

# construct an SVDD model
svdd = SVDD(parameters)

# train SVDD model
svdd.train(trainData, trainLabel)
# save the model to disk
filename = 'laplPC_model.sav'
pickle.dump(svdd, open(filename, 'wb'))
#svdd = pickle.load(open(filename, 'rb'))


tstart = datetime.now()
# test SVDD model
distance, accuracy, pred = svdd.test(testData, testLabel)
tend = datetime.now()
tspend = tend - tstart
print(tspend.seconds)

# visualize the results
draw.testResult(svdd, distance)
draw.testROC(testLabel, distance)
#draw.boundary(svdd, dur, trainLabel)
print('Accuracy: ', metrics.accuracy_score(testLabel, pred))
print('Precision: ', metrics.precision_score(testLabel, pred))
print('Recall: ', metrics.recall_score(testLabel, pred))
print('F1-Score: ', metrics.f1_score(testLabel, pred))
print('confusion_matrix: ', metrics.confusion_matrix(testLabel, pred))


#plt.scatter(testData[:, 0], testData[:, 1], c = distance, cmap = 'coolwarm', s = 2)