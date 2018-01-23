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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV

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

traindata = pd.read_csv('AnormlyDetection_2016_Train.csv', header=None)
#testdata = pd.read_csv('AnormlyDetection_2016_Train.csv', header=None)


X = traindata.iloc[:,0:242]
Y = traindata.iloc[:,243]
#C = testdata.iloc[:,243]
#T = testdata.iloc[:,0:242]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
print(trainX[0:5,:])

#scaler = Normalizer().fit(T)
#testT = scaler.transform(T)
# summarize transformed data
#np.set_printoptions(precision=3)
#print(testT[0:5,:])

y_train1 = np.array(Y)
#y_test1 = np.array(C)

y_train= to_categorical(y_train1)
#y_test= to_categorical(y_test1)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
print("processing")
grid_result = grid.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))



