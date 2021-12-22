import numpy as np
import pandas as pd
from glob import glob
import itertools
import os.path
import re
import tarfile
import time
import sys
import random
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.naive_bayes import MultinomialNB

# Loading dataset
#data = pd.read_csv('datset.csv')
partial_fit_classifiers = {
    'SGD': SGDClassifier(),
    'Perceptron': Perceptron(),
    'Passive-Aggressive': PassiveAggressiveClassifier(C=0.01, random_state=1),
}

scaler = StandardScaler()
filename = "New_DDOS_new.csv"
#n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
#s = 50000 #desired sample size
#skip = sorted(random.sample(range(1, n+2), n-s)) #the 0-indexed header will not be included in the skip list
df = pd.read_csv(filename)
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
df = df.drop(['Label_old'], axis=1)
df.fillna(df.mean(), inplace=True)
print(df.head())
y = df["Label"]
X = df.drop(["Label"], axis=1)
all_classes = np.array([0, 1])
# Splitting iris dataset into train and test sets
tick = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)
parsing_time = time.time() - tick
tick = time.time()
X_train = X_train.to_numpy().reshape(X_train.shape)
X_test = X_test.to_numpy().reshape(X_test.shape)
vectorizing_time = time.time() - tick
cls_stats = {}

for cls_name in partial_fit_classifiers:
    stats = {'n_train': 0, 'n_train_pos': 0,
             'accuracy': 0.0, 'accuracy_history': [(0, 0)], 't0': time.time(),
             'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
    cls_stats[cls_name] = stats
#y_train = y_train.to_numpy().reshape(-1, 1)
#y_test = y_test.to_numpy().reshape(-1, 1)

X_train = scaler.fit_transform(X_train)
#y_train = scaler.fit_transform(y_train)

X_test = scaler.fit_transform(X_test)
#y_test = scaler.fit_transform(y_test)
total_vect_time = 0.0
for cls_name, cls in partial_fit_classifiers.items():
    tick = time.time()
    # update estimator with examples in the current mini-batch
    cls.fit(X_train, y_train)

    # accumulate test accuracy stats
    cls_stats[cls_name]['total_fit_time'] += time.time() - tick
    cls_stats[cls_name]['n_train'] += X_train.shape[0]
    cls_stats[cls_name]['n_train_pos'] += sum(y_train)
    tick = time.time()
    cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
    cls_stats[cls_name]['prediction_time'] = time.time() - tick
    acc_history = (cls_stats[cls_name]['accuracy'],
                   cls_stats[cls_name]['n_train'])
    cls_stats[cls_name]['accuracy_history'].append(acc_history)
    run_history = (cls_stats[cls_name]['accuracy'], total_vect_time + cls_stats[cls_name]['total_fit_time'])
    cls_stats[cls_name]['runtime_history'].append(run_history)
# Creating model
def plot_accuracy(x, y, x_legend):
    """Plot accuracy as a function of x."""
    x = np.array(x)
    y = np.array(y)
    plt.title('Classification accuracy as a function of %s' % x_legend)
    plt.xlabel('%s' % x_legend)
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(x, y)

rcParams['legend.fontsize'] = 10
cls_names = list(sorted(cls_stats.keys()))

# Plot accuracy evolution
plt.figure()
for _, stats in sorted(cls_stats.items()):
    # Plot accuracy evolution with #examples
    accuracy, n_examples = zip(*stats['accuracy_history'])
    plot_accuracy(n_examples, accuracy, "training examples (#)")
    ax = plt.gca()
    ax.set_ylim((0.8, 1))
plt.legend(cls_names, loc='best')

plt.figure()
for _, stats in sorted(cls_stats.items()):
    # Plot accuracy evolution with runtime
    accuracy, runtime = zip(*stats['runtime_history'])
    plot_accuracy(runtime, accuracy, 'runtime (s)')
    ax = plt.gca()
    ax.set_ylim((0.8, 1))
plt.legend(cls_names, loc='best')

# Plot fitting times
plt.figure()
fig = plt.gcf()
cls_runtime = [stats['total_fit_time']
               for cls_name, stats in sorted(cls_stats.items())]

cls_runtime.append(total_vect_time)
cls_names.append('Vectorization')
bar_colors = ['b', 'g', 'r', 'c', 'm', 'y']

ax = plt.subplot(111)
rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5,
                     color=bar_colors)

ax.set_xticks(np.linspace(0, len(cls_names) - 1, len(cls_names)))
ax.set_xticklabels(cls_names, fontsize=10)
ymax = max(cls_runtime) * 1.2
ax.set_ylim((0, ymax))
ax.set_ylabel('runtime (s)')
ax.set_title('Training Times')


def autolabel(rectangles):
    """attach some text vi autolabel on rectangles."""
    for rect in rectangles:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.,
                1.05 * height, '%.4f' % height,
                ha='center', va='bottom')
        plt.setp(plt.xticks()[1], rotation=30)


autolabel(rectangles)
plt.tight_layout()
plt.show()

# Plot prediction times
plt.figure()
cls_runtime = []
cls_names = list(sorted(cls_stats.keys()))
for cls_name, stats in sorted(cls_stats.items()):
    cls_runtime.append(stats['prediction_time'])
cls_runtime.append(parsing_time)
cls_names.append('Read/Parse\n+Feat.Extr.')
cls_runtime.append(vectorizing_time)
cls_names.append('Hashing\n+Vect.')

ax = plt.subplot(111)
rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5,
                     color=bar_colors)

ax.set_xticks(np.linspace(0, len(cls_names) - 1, len(cls_names)))
ax.set_xticklabels(cls_names, fontsize=8)
plt.setp(plt.xticks()[1], rotation=30)
ymax = max(cls_runtime) * 1.2
ax.set_ylim((0, ymax))
ax.set_ylabel('runtime (s)')
ax.set_title('Prediction Times (%d instances)')
autolabel(rectangles)
plt.tight_layout()
plt.show()