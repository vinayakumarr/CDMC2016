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
from sklearn import metrics
from sklearn.metrics import (precision_score,   recall_score,
                             f1_score, accuracy_score, mean_squared_error, mean_absolute_error)


print("Loading")

traindata = pd.read_csv('normalized/anamoly_train.csv', header=None)
testdata = pd.read_csv('train_test.csv', header=None)


X = traindata.iloc[:,0:242]
Y = traindata.iloc[:,243]
C = testdata.iloc[:,243]
T = testdata.iloc[:,0:242]

X_train = np.array(X)
X_test = np.array(T)


y_train1 = np.array(Y)
y_test1 = np.array(C)

y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)


max_features = 500
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 8

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 10, input_length=maxlen))
model.add(LSTM(100))  # try using a GRU instead, for fun
model.add(Dense(8))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, batch_size, nb_epoch=1)
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))


#y_pred = model.predict_classes(X_test)
#np.savetxt('predictedlabelslstm.txt', np.transpose([y_test1,y_pred]), fmt='%01d')

'''
nmax = 2
n = 0

while n < nmax :
    y_pred = model.predict_classes(X_test)
    

    accuracy = accuracy_score(y_test1, y_pred)
    recall = recall_score(y_test1, y_pred , average="weighted")
    precision = precision_score(y_test1, y_pred , average="weighted")
    f1 = f1_score(y_test1, y_pred , average="weighted")
    mse = mean_squared_error(y_test1, y_pred)
    mae = mean_absolute_error(y_test1, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test1, y_pred,pos_label=2)
    auc = metrics.auc(fpr, tpr)

    print(mae)
    print('Accuracy: {}'.format(accuracy))
    print('Recall: {}'.format(recall))
    print('Precision: {}'.format(precision))
    print('F1: {}'.format(f1))
    print('Mean Squared Error')
    print(mse)
    print('Mean absolute Error')
    print(mae)
    print('Area under curve'.format(auc))
    print(metrics.classification_report(y_test1, y_pred))


print("***********************************")

'''
