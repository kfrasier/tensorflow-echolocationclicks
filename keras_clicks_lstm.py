import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam


# import Matlab data
import scipy.io as spio
mat = spio.loadmat('E:/Data/Papers/ClickClass2015/tensorflow/JAX13_disk5ab_trainSet.mat', squeeze_me=True)
x_train = mat['trainDataAll'] # array
y_train = keras.utils.to_categorical(mat['trainLabelsAll']) 
x_test = mat['testDataAll'] # array of structures
y_test = keras.utils.to_categorical(mat['testLabelsAll']) 

x_train = x_train[:,0:200]
x_test = x_test[:,0:200]

batch_size = 3000
x_train2 = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test2 = x_test.reshape(x_test.shape[0], x_test.shape[1],1)
y_train2 = y_train.reshape(y_train.shape[0],y_train.shape[1])
y_test2 = y_test.reshape(y_test.shape[0],y_test.shape[1])

input_shapeX = (x_train2.shape[1],x_train2.shape[2])
#input_shapeX = [300,1]
model = Sequential()
model.add(LSTM(200,
               input_shape=input_shapeX))  # returns a sequence of vectors of dimension 32
model.add(Dense(4, activation='softmax'))

opt = 'rmsprop'
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


model.fit(x_train2, y_train2,
          epochs=10,shuffle = True,
          batch_size=batch_size)

print("\nValidating ...")
score, accuracy = model.evaluate(x_test2, y_test2, batch_size=batch_size, verbose=1)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)

model.save('myModel_lstm.h5')
# model = load_model('myModel.h5')
#testOut = model.predict_classes(x_test2)
# mat = spio.savemat('testOut.mat', {'testOut':testOut})

