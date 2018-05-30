import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
# source ./bin/activate
# import Matlab data
import scipy.io as spio
mat = spio.loadmat('E:/Data/Papers/ClickClass2015/tensorflow/WAT_train_set1_singleCluster.mat', squeeze_me=True)
x_train = mat['x_train'] # array
y_train = keras.utils.to_categorical(mat['y_train']) 
x_test = mat['x_test'] # array of structures
y_test = keras.utils.to_categorical(mat['y_test']) 

batch_size = 300
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
#model.add(Dense(256, activation='relu', input_dim=300))
#model.add(Dropout(0.5))
model.add(Dense(176, activation='relu', input_dim=176))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

#sdg = SGD(lr=0.01, momentum=0.9)# decay=1e-6, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=50,shuffle= True,
          batch_size = batch_size)

print("\nValidating ...")
score, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)

model.save('E:/Data/Papers/ClickClass2015/tensorflow/WATModel_dense1_single.h5')
# model = load_model('myModel.h5')
testOut = model.predict_classes(x_test)
mat = spio.savemat('E:/Data/Papers/ClickClass2015/tensorflow/WAT_testOut1_single.mat', {'testOut':testOut})
