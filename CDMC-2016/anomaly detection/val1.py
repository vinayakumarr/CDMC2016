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

print("Loading")

traindata = pd.read_csv('AnormlyDetection_2016_Train.csv', header=None)
testdata = pd.read_csv('AnormlyDetection_2016_Train.csv', header=None)

x = traindata.iloc[:,0:242]
y = traindata.iloc[:,243]

scaler = Normalizer().fit(x)
trainX = scaler.transform(x)
# summarize transformed data
np.set_printoptions(precision=3)
print(trainX[0:5,:])


batch_size = 32

X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))



y_train1 = np.array(y)
y_train= to_categorical(y_train1)




model = Sequential()
print('Build model...')


def create_model():
  model = Sequential()
  model.add(LSTM(128, input_dim=242,dropout_W=0.1, dropout_U=0.1))
  model.add(Dense(8))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
  return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

model = KerasClassifier(build_fn=create_model, nb_epoch=10, batch_size=32)

# evaluate using 10-fold cross validation
kfold = StratifiedKFold(y=y_train1, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())




