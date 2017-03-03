import numpy as np
import h5py as h5
import os
import datetime 
from sklearn import preprocessing
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, RepeatVector
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed

os.chdir('/home/user3/work/')
	

batch_size = 32
nb_classes = 10
X_train = np.load('X_train1.npy')
X_test = np.load('X_test1.npy')
y_train = np.load('y_train1.npy')
y_test = np.load('y_test1.npy')

# with h5.File('data_aug1','r') as f1:
	# data1 = f1.get('video_train')[:].transpose()
	# data2 = f1.get('audio_train')[:].transpose()
	# data1 = np.array(data1)
	# data2 = np.array(data2)
	# X_train = np.hstack((data1,data2))
	
# with h5.File('data_test.h5','r') as f3:
     # data2 = f3.get('video_test')[:].transpose()
     # data3 = f3.get('audio_test')[:].transpose()
     # data2 = np.array(data2)
     # data3 = np.array(data3)
     # X_test = np.hstack((data2,data3))
# X_train = X_train.reshape(960,40,300)
# X_test = X_test.reshape(60,40,300)

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
print(X_train.shape)
print(X_test.shape)
model1 = Sequential()
model11 = Sequential()
#model11.add(TimeDistributed(Dense(250,activation="relu"),input_shape=X_train.shape[1:]))
model11.add(GRU(output_dim=200, return_sequences=True, activation='sigmoid',input_shape=X_train.shape[1:]))  #,input_shape=X_train.shape[1:]
model1.add(model11)
model12 = Sequential()
model12.add(GRU(output_dim=300, return_sequences=True,input_shape = (40,200), activation='sigmoid'))#, activation='sigmoid'
#model12.add(TimeDistributed(Dense(300,activation="sigmoid")))
model1.add(model12)

# model1 = Sequential()
# model11=Sequential()
# mode112=Sequential()
# model11.add(Dense(250,input_dim=300))
# model11.add(Activation('relu'))
# model11.add(Dense(200))
# model11.add(Activation('sigmoid'))
# model1.add(model11)
# model12.add(Dense(250))
# model12.add(Activation('relu'))
# model12.add(Dense(300))
# model12.add(Activation('sigmoid'))
# model1.add(model12)

# inputs = Input(shape=(40,300))
# encoded = LSTM(200)(inputs)
# decoded = RepeatVector(40)(encoded)
# decoded = LSTM(300, return_sequences=True)(decoded)
# model1 = Model(inputs, decoded)
# model11 = Model(inputs, encoded)

model1.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['binary_crossentropy'])
model1.fit(X_train, X_train, batch_size=batch_size, nb_epoch=10,validation_data=(X_test,X_test))

with h5.File('data_vae2','r') as f:
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

'''
reshape 960*40*200  and 60*40*200

'''
# X_train2 = X_train1.reshape(16,2400,300)
# X_test2 = X_test1.reshape(1,2400,300)

temp1 = model11.predict(X_train2)
temp2 = model11.predict(X_test2)

# temp1 = temp1.squeeze()
# temp2 = temp2.squeeze()

# temp1 = temp1.reshape(960,40,200)
# temp2 = temp2.reshape(60,40,200)

with h5.File('data_lstmae1','w') as f:
	f['train'] = temp1
	f['test'] = temp2

# with h5.File('data_lstmae','r') as f:
	# temp1 = f['train'][:]
	# temp2 = f['test'][:]


print('-----------------------------')

model2 = Sequential()
model21 = Sequential()
model21.add(LSTM(output_dim=150, return_sequences=True,input_shape=temp1.shape[1:],dropout_W=0.3))#W_regularizer=l2(l=0.01)
model21.add(LSTM(output_dim=100, return_sequences=True))
model21.add(LSTM(50))  #,W_regularizer=l2(l=0.01)  ,dropout_W=0.1
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

