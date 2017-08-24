from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
import pandas as pa
import numpy as np
from numpy import argmax

word_max_features = 1116
maxlen = 29  # cut texts after this number of words (among top max_features most common words)
batch_size = 20
train_data_df= pa.read_csv('sequenceData/trainSentenceSequence.txt',header=None)
train_label_df=pa.read_csv('sequenceData/trainLabelSequence.txt',header=None)
test_data_df= pa.read_csv('sequenceData/testSentenceSequence.txt',header=None)
test_label_df=pa.read_csv('sequenceData/testLabelSequence.txt',header=None)

x_train = np.array(train_data_df.iloc[:,0:29])
y_train = np.array(train_label_df.iloc[:,0:8])
x_test = np.array(test_data_df.iloc[:,0:29])
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
model = Sequential()
model.add(Embedding(word_max_features, 128))
model.add(LSTM(256,return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(32, activation='sigmoid'))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(8, activation='softmax'))

# try using different optimizers and different optimizer configs

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')

#model.fit(x_train, y_train,batch_size=batch_size,validation_split=0.1,epochs=40)


#model.save('model/test.mod')

#model=load_model('model/test.mod')

score, acc = model.evaluate(x_test, y_test)
print('Test score:', score)
print('Test accuracy:', acc)

for i,data in enumerate(x_test):
	res = model.predict(data)
	print (argmax(res))
	#print(res.shape)