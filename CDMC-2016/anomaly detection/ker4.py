import keras.preprocessing.text
import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import Normalizer
from keras.layers import Dense, Dropout, Activation, Embedding

print("Loading")

traindata = pd.read_csv('anomaly1/traina.csv', header=None)
testdata = pd.read_csv('anomaly1/train_test.csv', header=None)


X = traindata.iloc[:,0:244]
Y = traindata.iloc[:,243]
C = testdata.iloc[:,243]
T = testdata.iloc[:,0:244]


scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
#print(trainX[0:5,:])

scaler = Normalizer().fit(T)
testT = scaler.transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])


y_train1 = np.array(Y)
y_test1 = np.array(C)

y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)




# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))

batch_size=16

model = Sequential()
model.add(LSTM(256, input_dim=244, return_sequences=True))
model.add(Dropout(0.9))
model.add(LSTM(256, input_dim=244, return_sequences=False))
model.add(Dropout(0.5))
#model.add(LSTM(256, input_dim=244, return_sequences=True))
#model.add(Dropout(0.5))
#model.add(LSTM(256, input_dim=244, return_sequences=False))
#model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size, nb_epoch=10, validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test, batch_size=1)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))


y_pred = model.predict_classes(X_test)
np.savetxt('predictedlabelsnew.txt', np.transpose([y_test1,y_pred]), fmt='%01d')



