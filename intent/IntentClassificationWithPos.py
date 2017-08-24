from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,Concatenate
from keras.layers import Merge
from keras.datasets import imdb
import pandas as pa
import numpy as np


word_max_features = 1116
pos_max_features=42
maxlen = 29  # cut texts after this number of words (among top max_features most common words)
batch_size = 20
train_data_df= pa.read_csv('sequenceData/trainSentenceSequence.txt',header=None)
train_label_df=pa.read_csv('sequenceData/trainLabelSequence.txt',header=None)
train_pos_df=pa.read_csv('sequenceData/trainPosSequence.txt',header=None)
test_data_df= pa.read_csv('sequenceData/testSentenceSequence.txt',header=None)
test_label_df=pa.read_csv('sequenceData/testLabelSequence.txt',header=None)
test_pos_df=pa.read_csv('sequenceData/testPosSequence.txt',header=None)

x_train = np.array(train_data_df.iloc[:,0:29])
p_train = np.array(train_pos_df.iloc[:,0:29])
y_train = np.array(train_label_df.iloc[:,0:8])

x_test = np.array(test_data_df.iloc[:,0:29])
p_test = np.array(test_pos_df.iloc[:,0:29])
y_test = np.array(test_label_df.iloc[:,0:8])
"""
train_test_df= pa.read_csv('test_sequence.csv',header=None)
label_test_df=pa.read_csv('test_label.csv',header=None)
print('Loading data...')




print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=8)
"""
print('x_train shape:', x_train.shape)

print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)

print('y_test shape:', y_test.shape)
print('Build model...')

word_model = Sequential()
word_model.add(Embedding(word_max_features, 128))

pos_model=Sequential()
pos_model.add(Embedding(pos_max_features,128))

merged=Merge([word_model,pos_model])
model=Sequential()
model.add(merged)

model.add(LSTM(256,return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(32, activation='sigmoid'))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(8, activation='sigmoid'))
# try using different optimizers and different optimizer configs

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')

model.fit([x_train,p_train], y_train,
          batch_size=batch_size,validation_split=0.1,
          epochs=25,
          )

score, acc = model.evaluate([x_test,p_test], y_test)
print('Test score:', score)
print('Test accuracy:', acc)

