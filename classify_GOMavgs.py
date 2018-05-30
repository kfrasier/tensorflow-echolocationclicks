import keras
from keras.models import load_model
import scipy.io as spio
import os
import h5py
import numpy as np

fileName = 'E:/Data/Papers/ClickClass2015/tensorflow/GOM_classify_GC.mat'
matData = spio.loadmat(fileName, squeeze_me=True)
x_train = matData['toClassify'] 

model = load_model('E:/Data/Papers/ClickClass2015/tensorflow/myModel_dense_2000.h5')
outDir = 'E:/Data/Papers/ClickClass2015/tensorflow/'
print(x_train.shape)
predictedLabels = model.predict_classes(x_train)
probs = model.predict(x_train)
mat = spio.savemat('E:/Data/Papers/ClickClass2015/tensorflow/predictedLabels_2000_GC.mat', 
	{'predictedLabels':predictedLabels,'probs':probs})
print('Done classifying.')
