import keras
import h5py
import hdf5storage

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import regularizers
from keras.optimizers import SGD
import numpy as np
#source ./bin/activate
# import Matlab data
import scipy.io as spio

trainSetFile = 'J:/DCL/forNNet_wICI/nnet_DCL2015_train_norm.mat'
mat1 = h5py.File(trainSetFile, 'r')

x_train = mat1['trainDataAll'].value
x_train = x_train.transpose()
y_trainMat = mat1['trainLabelsAll'].value
print(x_train.shape)
print(y_trainMat.shape)


testSetFile = 'J://DCL/forNNet_wICI/nnet_DCL2015_test_norm.mat'
mat2 = h5py.File(testSetFile, 'r')
x_test = mat2['testDataAll'].value
x_test = x_test.transpose()
y_testMat = mat2['testLabelsAll'].value

y_train = keras.utils.to_categorical(y_trainMat.transpose()-1)
y_test = keras.utils.to_categorical(y_testMat.transpose()-1)

#x_train = x_train[:,0:291]
#x_test = x_test[:,0:291]
print(x_train.shape)
print(y_train.shape)

batch_size = 5000
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
#model.add(Dense(291, activation='relu', input_dim=421))
#model.add(Dropout(0.5))
model.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.001), input_dim=291))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

#sdg = SGD(lr=0.01, momentum=0.9)# decay=1e-6, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,shuffle= True,
          batch_size = batch_size)

print("\nValidating ...")
score, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)

model.save('J:/DCL/forNNet_wICI/DCL_noICI_testOut.h5')
# model = load_model('myModel.h5')
testOut = model.predict_classes(x_test)
probs = model.predict(x_test)

mat = spio.savemat('J:/DCL/forNNet_wICI/DCL_noICI_testOut.mat',
	{'testOut':testOut,'probs':probs})


# model = load_model('myModel_dense_unsup.h5')