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

print("Loading")

traindata = pd.read_csv('normalized/anamoly_train.csv', header=None)
testdata = pd.read_csv('normalized/anamoly_test.csv', header=None)

x = traindata.iloc[:,0:242]
y = traindata.iloc[:,243]

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=42)


X_train=np.array(X_train)
X_test=np.array(X_test)


maxfeatures = 242
maxlen = 200  # cut texts after this number of words (among top max_features most common words)
batch_size = 32



X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)


y_train1 = np.array(y_train)
y_test1 = np.array(y_test)

y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)


max_features = 200
model = Sequential()
print('Build model...')

model = Sequential()
model.add(Embedding(maxfeatures, 128, input_length=maxlen, dropout=0.1))
model.add(LSTM(128, dropout_W=0.1, dropout_U=0.1))
model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
model.save_weights("model.h5")
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
