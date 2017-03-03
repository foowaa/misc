import numpy as np
import h5py as h5
from sklearn import manifold

import os
import datetime 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Reshape
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed


os.chdir('/home/user3/work')
with h5.File('data_aug1','r') as f1:
	video_train = f1['video_train'][:].transpose()
	audio_train = f1['audio_train'][:].transpose()
	label_train = f1['label_train'][:].transpose()
	label_train = label_train.squeeze().astype(int)
	#label_train = label_train.astype(int)

with h5.File('data_test.h5','r') as f2:
	video_test = f2['video_test'][:].transpose()
	audio_test = f2['audio_test'][:].transpose()
	label_test = f2['label_test'][:].transpose()
	label_test = label_test.squeeze().astype(int)

print('------')

'''
http://scikit-learn.org/stable/modules/manifold.html
'''
# audio_ = np.vstack((audio_train,audio_test))
# lle = manifold.LocallyLinearEmbedding(3,100)
# audio_lle = lle.fit_transform(audio_)
# audio_train_lle = audio_lle[:38400,:]
# audio_test_lle = audio_lle[38400:,:]
# np.save('audio_train_lle',audio_train_lle)
# np.save('audio_test_lle', audio_test_lle)
X_train = np.hstack((video_train,audio_train))
X_test = np.hstack((video_test, audio_test))
y_train = label_train
y_test = label_test

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
'''
define models


'''
model = Sequential()

mlp = Sequential()
mlp.add(Dense(250, input_dim=300))
mlp.add(Activation('relu'))
mlp.add(Dense(200))
mlp.add(Activation('relu'))
mlp.add(Reshape((,40,200)))
model.add(mlp)

lstm = Sequential()

lstm.add(LSTM(output_dim=150, return_sequences=True, input_shape=(40,200)))
lstm.add(LSTM(output_dim=100, return_sequences=True)) #,dropout_W=0.1
lstm.add(LSTM(50))
model.add(lstm)

clas = Sequential()
clas.add(Dense(30,input_dim=50))
clas.add(Activation('relu'))
clas.add(BatchNormalization(mode=2))
clas.add(Dense(20))
clas.add(Activation('relu'))
clas.add(BatchNormalization(mode=2))
clas.add(Dense(nb_classes))
clas.add(Activation('softmax'))
model.add(clas)




model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, nb_epoch=3000, validation_data=(X_test,y_test))