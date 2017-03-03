import numpy as np
import h5py 
import os
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

os.chdir('/home/user3/work/py/')

with h5py.File('result_torch.h5','r') as f1:
	data = f1.get('data')
	data = np.array(data)
	#data1 = data.transpose()
	q = np.zeros((480,40,100))
	j = 0
	p = []
	for i in xrange(0,19200):
		d = data[i,]
		d = d[np.newaxis,]
		if (i+1)%40 == 1:
			p.append(d)
		elif (i+1)%40 == 0:
			p[j] = np.vstack((p[j],d))
			q[j,...] = p[j]
			j += 1
		else:
			p[j] = np.vstack((p[j],d))



with h5py.File('data_train.h5','r') as f2:
	label = f2.get('label_train')
	label = np.array(label)
	label1 = np.squeeze(label)
	i = 0
	labelx = np.zeros(480)
	for j in xrange(39,19200,40):
		labelx[i] = label1[j]
		i += 1
	labelx = labelx.astype(int)
'''
final data are q and labelx

'''
batch_size = 2
hidden_units = 100
nb_classes = 10
X_train = q
y_train = labelx

y_train = np_utils.to_categorical(y_train, nb_classes)

model = Sequential()
model.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
	           forget_bias_init='one', activation='tanh', inner_activation='sigmoid',
	           input_shape=X_train.shape[1:])) 
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10)
