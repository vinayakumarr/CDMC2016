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
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau


print("Loading")

traindata = pd.read_csv('eNews_2016_Train.csv', header=None)
testdata = pd.read_csv('eNews_2016_Test.csv', header=None)


x = traindata.iloc[:,0]
y = traindata.iloc[:,1]
t = testdata.iloc[:,0]


tk = keras.preprocessing.text.Tokenizer(nb_words=5000, filters=keras.preprocessing.text.base_filter(), lower=True, split=" ")
tk.fit_on_texts(x)

x = tk.texts_to_sequences(x)


tk = keras.preprocessing.text.Tokenizer(nb_words=5000, filters=keras.preprocessing.text.base_filter(), lower=True, split=" ")
tk.fit_on_texts(t)
t = tk.texts_to_sequences(t)

y = np.array(y)

max_len = 500
print "max_len ", max_len
print('Pad sequences (samples x time)')

x = sequence.pad_sequences(x, maxlen=max_len)
t = sequence.pad_sequences(t, maxlen=max_len)
y= to_categorical(y)

max_features = 5000

model = Sequential()
print('Build model...')

model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len, dropout=0.1))
model.add(LSTM(128, dropout_W=0.1, dropout_U=0.1))
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('logs/training_set_iranalysis.csv',separator=',', append=False)
model.fit(x, y, batch_size=32, nb_epoch=1000, validation_data=(x, y),callbacks=[checkpointer,csv_logger])
y_pred = model.predict_classes(t)

np.savetxt('output.txt', y_pred, fmt='%01d')


score, acc = model.evaluate(x, y,batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

