import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D


# import Matlab data
import scipy.io as spio
mat = spio.loadmat('E:/Data/Papers/ClickClass2015/tensorflow/JAX13_disk5ab_trainSet.mat', squeeze_me=True)
x_train = mat['trainDataAll'] # array
y_train = keras.utils.to_categorical(mat['trainLabelsAll']) 
x_test = mat['testDataAll'] # array of structures
y_test = keras.utils.to_categorical(mat['testLabelsAll']) 

x_train = x_train[:,0:200]
x_test = x_test[:,0:200]

x_train2 = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test2 = x_test.reshape(x_test.shape[0], x_test.shape[1],1)
y_train2 = y_train.reshape(y_train.shape[0],y_train.shape[1])
y_test2 = y_test.reshape(y_test.shape[0],y_test.shape[1])

batch_size = 5000

model = Sequential()
model.add(Conv1D(100, 2, activation='relu', input_shape=(200, 1)))
model.add(MaxPooling1D(3))
model.add(Conv1D(25, 4, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train2, y_train2,
          epochs=30,
          batch_size = batch_size)

print("\nValidating ...")
score, accuracy = model.evaluate(x_test2, y_test2, batch_size=batch_size, verbose=1)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)

model.save('E:/Data/Papers/ClickClass2015/tensorflow/JAX13_disk5ab_myModel_conv.h5')
testOut = model.predict_classes(x_test2)
testOut = model.predict_classes(x_test2)
mat = spio.savemat('E:/Data/Papers/ClickClass2015/tensorflow/JAX13_disk5ab_testOut_conv.mat',
	{'testOut':testOut})
