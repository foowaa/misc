import numpy as np
import h5py 
import os
import datetime 

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Lambda
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input, Dense, merge
from keras.models import Model

from keras import backend as K

start = datetime.datetime.now()


'''
preprocessing
'''

os.chdir('/home/user3/work/')
number_of_video_train = 480*2
number_of_video_test = 60
frames_of_every_video = 40
number_of_attr_v = 100
number_of_attr_a = 100
number_of_train = 19200*2
number_of_test = 2400

with h5py.File('data_cnn_aug1.h5','r') as f1:       #data_cnn_aug1.h5
	data1 = f1.get('video_train')
	data1 = np.array(data1)
	data1 = data1.transpose()
	print(data1.shape)
	q1 = np.zeros((number_of_video_train,frames_of_every_video,number_of_attr_v))
	j = 0
	p = []
	for i in xrange(0,number_of_train):
		d = data1[i,]
		d = d[np.newaxis,]
		if (i+1)%frames_of_every_video == 1:
			p.append(d)
		elif (i+1)%frames_of_every_video == 0:
			p[j] = np.vstack((p[j],d))
			q1[j,...] = p[j]
			j += 1
		else:
			p[j] = np.vstack((p[j],d))
   
with  h5py.File('data_cnn.h5','r') as f4:          #data_cnn.h5
	data2 = f4.get('video_test')
	data2 = np.array(data2)
	data2 = data2.transpose()
	print(data2.shape)
	q2 = np.zeros((number_of_video_test,frames_of_every_video,number_of_attr_v))
	j = 0
	p = []
	for i in xrange(0,number_of_test):
		d = data2[i,]
		d = d[np.newaxis,]
		if (i+1)%frames_of_every_video == 1:
			p.append(d)
		elif (i+1)%frames_of_every_video == 0:
			p[j] = np.vstack((p[j],d))
			q2[j,...] = p[j]
			j += 1
		else:
			p[j] = np.vstack((p[j],d))

with  h5py.File('data_daex','r') as f4:       #data_daex
	data2 = f4.get('audio_train')
	data2 = np.array(data2)
	#data2 = data2.transpose()
	print(data2.shape)
	q3 = np.zeros((number_of_video_train,frames_of_every_video,number_of_attr_a))
	j = 0
	p = []
	for i in xrange(0,number_of_test):
		d = data2[i,]
		d = d[np.newaxis,]
		if (i+1)%frames_of_every_video == 1:
			p.append(d)
		elif (i+1)%frames_of_every_video == 0:
			p[j] = np.vstack((p[j],d))
			q3[j,...] = p[j]
			j += 1
		else:
			p[j] = np.vstack((p[j],d))

with  h5py.File('data_daex','r') as f4:   #data_daex
	data2 = f4.get('audio_test')
	data2 = np.array(data2)
	#data2 = data2.transpose()
	print(data2.shape)
	q4 = np.zeros((number_of_video_test,frames_of_every_video,number_of_attr_a))
	j = 0
	p = []
	for i in xrange(0,number_of_test):
		d = data2[i,]
		d = d[np.newaxis,]
		if (i+1)%frames_of_every_video == 1:
			p.append(d)
		elif (i+1)%frames_of_every_video == 0:
			p[j] = np.vstack((p[j],d))
			q4[j,...] = p[j]
			j += 1
		else:
			p[j] = np.vstack((p[j],d))


with h5py.File('data_cnn_aug1.h5','r') as f2:    #data_cnn_aug1.h5
	label = f2.get('label_train')
	label = np.array(label)
	label1 = np.squeeze(label)
	i = 0
	labelx = np.zeros(number_of_video_train)
	for j in xrange(39,number_of_train,40):
		labelx[i] = label1[j]
		i += 1
	labelx = labelx.astype(int)


with h5py.File('data_cnn.h5','r') as f3:     #data_cnn.h5
	label = f3.get('label_test')
	label = np.array(label)
	label2 = np.squeeze(label)
	i = 0
	labely = np.zeros(number_of_video_test)
	for j in xrange(39,number_of_test,40):
		labely[i] = label2[j]
		i += 1
	labely = labely.astype(int)


'''
final data are q and labelx

'''
print('Building LSTM Model...')
batch_size = 32

nb_classes = 10
video_train = q1
y_train = labelx
video_test = q2
y_test = labely
audio_train = q3
audio_test = q4


print(video_train.shape)
print(audio_train.shape)
print(video_test.shape)
print(audio_test.shape)
print(y_train.shape)
print(y_test.shape)
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

print('------------------')

'''
define model
'''
audio_test1 = np.zeros((60, 40, 100))
video_test1 = np.zeros((60, 40, 100))
video_train1 = np.zeros((960,40,100))

input1 = Input(shape=video_train.shape[1:])
input2 = Input(shape=audio_train.shape[1:])
lstm1 = LSTM(output_dim = 50, return_sequences=True, dropout_W=0.2,dropout_U=0.2)(input1)
lstm2 = LSTM(output_dim = 50, return_sequences=True, dropout_W=0.2,dropout_U=0.2)(input2)
#----
lambda_out1 = Lambda(lambda x:K.sum(x,1))(lstm1)
lambda_out2 = Lambda(lambda x:K.sum(x,1))(lstm2)
aux_out1 = Dense(10, activation='softmax',name='aux_out1')(lambda_out1)
aux_out2 = Dense(10, activation='softmax', name='aux_out2')(lambda_out2)
#----
merge1 = merge([lstm1, lstm2], mode='sum')
merge1_act = Activation('tanh')(merge1)
merge1_out = Lambda(lambda x:K.sum(x,1))(merge1_act)
clas1 = Dense(30, activation='relu')(merge1_out)
bn1 = BatchNormalization(mode=2)(clas1)
clas2 = Dense(20, activation='relu')(bn1)
bn2 = BatchNormalization(mode=2)(clas2)
clas3 = Dense(10, activation='softmax',name='clas3')(bn2)


mlstm = Model(input=[input1, input2], output=[clas3, aux_out1, aux_out2])

mlstm.compile(optimizer='rmsprop', 
			  loss={'clas3':'categorical_crossentropy', 'aux_out1':'categorical_crossentropy', 'aux_out2':'categorical_crossentropy'},
			  loss_weights={'clas3':1., 'aux_out1': 0.05, 'aux_out2':0.1},
			  metrics={'clas3':['accuracy']})
mlstm.fit([video_train, audio_train],[y_train,y_train,y_train], nb_epoch=3000, batch_size=32, validation_data=([[video_test1, audio_test],[y_test,y_test,y_test]]))

