import socket
import numpy as np
import pandas as pd
import socket                   # Import socket module
import pickle
import time
import random
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

scaler = StandardScaler()
filename = "New_DDOS_new.csv"
df = pd.read_csv(filename)
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
df = df.drop(['Label_old'], axis=1)
df.fillna(df.mean(), inplace=True)
#print(df.head())
y = df["Label"]
X = df.drop(["Label"], axis=1)

# Splitting iris dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)

#host = '192.168.1.1' #L'IP du Serveur
#port = 1234
#server.connect((host,port))

#msg =server.recv(1024)
#print(msg)
#'Client Online ...'.encode('UTF-8')
#server.send(X_test)
#input()

# client.py
import socket, pickle

HOST = 'localhost'
PORT = 50007
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
arr = X_test.to_numpy()
data_string = pickle.dumps(arr)
s.send(data_string)

data = s.recv(4096)
data_arr = pickle.loads(data)
s.close()
print('Received', repr(data_arr))