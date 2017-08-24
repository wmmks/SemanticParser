from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential

from keras.layers import Dense, Embedding,Dropout
from keras.layers import LSTM, Bidirectional
from keras.layers import Masking
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Merge
from numpy import argmax
import pandas as pa
import numpy as np


def loadData(fileName):
	data= open(fileName,'r')
	return data

def appendLabel(data):
	allLabel=[]
	labelDic={}
	size=0
	max_leng=0
	for lines in data.readlines():
		size+=1
		seq=[]
		if len(lines.split(',')) > max_leng:
			max_leng=len(lines.split(','))
		for lab in lines.split(","):
		
			if lab=='\r\n':
				continue
			labelDic[lab]=len(labelDic)+1
		
			seq.append(float(lab))
		
		allLabel.append(seq)
	return allLabel,labelDic,size,max_leng
	
def prepareLabel(size,max_leng,label_size,allLabel):
	labelarray=np.zeros((size,max_leng,label_size+1),dtype=np.float)
	
	for i,data in enumerate(allLabel):
		tmp=np.array(np_utils.to_categorical(data,label_size+1))
		labelarray[i,:tmp.shape[0],:tmp.shape[1]]=tmp
	return labelarray

def appendTrainData(data):
	max_feature_leng=0
	allTrain=[]
	size=0
	max_leng=0
	for data in train_data.readlines():
		trainSeq=[]
		size+=1
		if len(data.split(','))>max_leng:
			max_leng=len(data.split(','))
		for i,feature in enumerate(data.split(',')):
			if feature=='\r\n':
				continue;
			trainSeq.append(float(feature))
			if float(feature)>max_feature_leng:
				max_feature_leng=float(feature)
		allTrain.append(trainSeq)
	return allTrain,size,max_leng,max_feature_leng
	
def prepareTrainData(allTrain):
	trainData=np.zeros((size,max_leng),dtype=np.float)
	for i,data in enumerate(allTrain):
		trainData[i,:len(data)]=data
	return trainData
	
train_size=394
test_size=42
max_leng=15
max_features=877
max_pos_features=43
max_dependency_features=38
label_size=26
train_data_df= pa.read_csv('SequenceData/trainWordSequence.txt',header=None)
train_pos_df= pa.read_csv('SequenceData/trainPosSequence.txt',header=None)
train_dep_df= pa.read_csv('SequenceData/trainDependencySequence.txt',header=None)
train_label_df=pa.read_csv('SequenceData/trainFilteredSequentialLabel.txt',header=None)
test_data_df= pa.read_csv('SequenceData/testWordSequence.txt',header=None)
test_pos_df= pa.read_csv('SequenceData/testPosSequence.txt',header=None)
test_dep_df= pa.read_csv('SequenceData/testDependencySequence.txt',header=None)
test_label_df=pa.read_csv('SequenceData/testFilteredSequentialLabel.txt',header=None)
x_train = np.array(train_data_df.iloc[:,0:15])
p_train = np.array(train_pos_df.iloc[:,0:15])
d_train = np.array(train_dep_df.iloc[:,0:15])
y_train = np.array(train_label_df.iloc[:,0:15])
x_test = np.array(test_data_df.iloc[:,0:15])
p_test = np.array(test_pos_df.iloc[:,0:15])
d_test = np.array(test_dep_df.iloc[:,0:15])
y_test = np.array(test_label_df.iloc[:,0:15])

y_train=prepareLabel(train_size,max_leng,label_size,y_train)
y_test=prepareLabel(test_size,max_leng,label_size,y_test)
print(x_train.shape,y_train.shape)

batch_size=30


word_model = Sequential()
word_model.add(Embedding(int(max_features), 128,mask_zero=True))

pos_model = Sequential()
pos_model.add(Embedding(int(max_pos_features),128,mask_zero=True))

merged=Merge([word_model,pos_model],mode='concat')
final_model = Sequential()
final_model.add(merged)
word_model.add(Dropout(0.2))
final_model.add(LSTM(256,return_sequences=True))
word_model.add(TimeDistributed(Dense(27, activation='softmax')))
final_model.add(Dense(27,activation='softmax'))
final_model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
              metrics=['categorical_accuracy'])
			  
		  
final_model.fit([x_train,p_train],y_train,batch_size=batch_size,validation_split=0.1,epochs=500)
		  

final_model.save_weights('model/WP.mod')

final_model.load_weights('model/WP.mod')

score, acc = final_model.evaluate([x_test,p_test], y_test,batch_size=batch_size)


print('Test score:', score)
print('Test accuracy:', acc)
