from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from keras.optimizers import SGD


traindata = pd.read_csv('anomaly/train.csv', header=None)
testdata = pd.read_csv('anomaly/test.csv', header=None)


X = traindata.iloc[:,0:244]
Y = traindata.iloc[:,0]
C = testdata.iloc[:,0]
T = testdata.iloc[:,0:244]

scaler = Normalizer().fit(X)
X_train = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)

scaler = Normalizer().fit(T)
X_test = scaler.transform(T)
# summarize transformed data
np.set_printoptions(precision=3)


y_train1 = np.array(Y)
y_test1 = np.array(C)

y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)

batch_size=16
nb_epoch =20

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(256, input_dim=244, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(10))
model.add(Dense(256, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.9))
model.add(Dense(8, init='uniform'))
model.add(Dropout(0.9))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=100)

loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_pred = model.predict_classes(X_test)
np.savetxt('outputmlp.txt', np.transpose([y_test1,y_pred]), fmt='%01d')

