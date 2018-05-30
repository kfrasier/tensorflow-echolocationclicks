import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
#source ./bin/activate
# import Matlab data
import scipy.io as spio
mat = spio.loadmat('E:/Data/Papers/ClickClass2015/tensorflow/JAX13_unsupervisedTS_set.mat', squeeze_me=True)
x_train = mat['trainDataAll'] # array
y_train = keras.utils.to_categorical(mat['trainLabelsAll']-1)
x_test = mat['testDataAll'] # array of structures
y_test = keras.utils.to_categorical(mat['testLabelsAll']-1)

x_train = x_train[:,0:200]
x_test = x_test[:,0:200]

batch_size = 5000
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
#model.add(Dense(256, activation='relu', input_dim=300))
#model.add(Dropout(0.5))
model.add(Dense(200, activation='relu', input_dim=200))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

#sdg = SGD(lr=0.01, momentum=0.9)# decay=1e-6, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=100,shuffle= True,
          batch_size = batch_size)

print("\nValidating ...")
score, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)

model.save('myModel_dense_unsup.h5')
# model = load_model('myModel.h5')
testOut = model.predict_classes(x_test)
mat = spio.savemat('E:/Data/Papers/ClickClass2015/tensorflow/JAX13_disk_unsup_testOut.mat', {'testOut':testOut})
