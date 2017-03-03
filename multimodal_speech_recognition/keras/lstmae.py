import numpy as np
import h5py as h5
import os
import datetime 
from sklearn import preprocessing
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
os.chdir('/home/user3/work/')
batch_size = 32
nb_classes = 10
X_train = np.load('X_train2.npy')
X_test = np.load('X_test2.npy')
y_train = np.load('y_train2.npy')
y_test = np.load('y_test2.npy')





y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model1 = Sequential()
model11 = Sequential()
model11.add(LSTM(output_dim=200, return_sequences=True,input_shape=X_train.shape[1:],activation='sigmoid'))
model1.add(model11)
model12 = Sequential()
model12.add(LSTM(output_dim=300, return_sequences=True,input_shape = (40,200), activation='sigmoid'))
model1.add(model12)
model1.compile(loss='binary_crossentropy', optimizer = 'rmsprop', metrics=['binary_crossentropy'])
model1.fit(X_train, X_train, batch_size=batch_size, nb_epoch=80, validation_data=(X_test,X_test))

with h5.File('data_vae1','r') as f:
	X_train1 = f['train'][:]
	X_test1 = f['test'][:]
	

number_of_video_train = 480*2
number_of_video_test = 60
frames_of_every_video = 40
number_of_attr = 300
number_of_train = 19200*2
number_of_test = 2400

q1 = np.zeros((number_of_video_train,frames_of_every_video,number_of_attr))
j = 0
p = []
for i in xrange(0,number_of_train):
	d = X_train1[i,]
	d = d[np.newaxis,]
	if (i+1)%frames_of_every_video == 1:
		p.append(d)
	elif (i+1)%frames_of_every_video == 0:
		p[j] = np.vstack((p[j],d))
		q1[j,...] = p[j]
		j += 1
	else:
		p[j] = np.vstack((p[j],d))
			


q2 = np.zeros((number_of_video_test,frames_of_every_video,number_of_attr))
j = 0
p = []
for i in xrange(0,number_of_test):
	d = X_test1[i,]
	d = d[np.newaxis,]
	if (i+1)%frames_of_every_video == 1:
		p.append(d)
	elif (i+1)%frames_of_every_video == 0:
		p[j] = np.vstack((p[j],d))
		q2[j,...] = p[j]
		j += 1
	else:
		p[j] = np.vstack((p[j],d))
	
X_train2 = q1
X_test2 = q2	


temp1 = model11.predict(X_train2)
temp2 = model11.predict(X_test2)

with h5.File('data_lstmae1','w') as f:
	f['train'] = temp1
	f['test'] = temp2

# with h5.File('data_lstmae','r') as f:
	# temp1 = f['train'][:]
	# temp2 = f['test'][:]


print('-----------------------------')

model2 = Sequential()
model21 = Sequential()
model21.add(LSTM(output_dim=150, return_sequences=True,input_shape=temp1.shape[1:],W_regularizer=l1(l=0.01)))
model21.add(LSTM(output_dim=100, return_sequences=True))
model21.add(LSTM(50,W_regularizer=l2(l=0.01)))  #,W_regularizer=l2(l=0.01)
model2.add(model21)

model22 = Sequential()
model22.add(Dense(30,input_dim=50))
model22.add(Activation('relu'))
model22.add(BatchNormalization(mode=2))
model22.add(Dense(20))
model22.add(Activation('relu'))
model22.add(BatchNormalization(mode=2))

model22.add(Dense(nb_classes))
model22.add(Activation('softmax'))
model2.add(model22)

model2.compile(loss='categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
model2.fit(temp1, y_train, batch_size=batch_size, nb_epoch=3000, validation_data=(temp2,y_test))

