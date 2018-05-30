import keras
from keras.models import load_model
import scipy.io as spio
import os
import h5py
import hdf5storage
import numpy as np

def load_ts(fileName):
	matData = spio.loadmat(fileName, squeeze_me=True)
	x_train = mat['trainDataAll'] # array
	return x_train


directory = os.fsencode('D:/WAT_HZmetadata/TPWS/ClusterBins_120dB_linear')
# load trained network
os.chdir(directory)
model = load_model('E:/Data/Papers/ClickClass2015/tensorflow/WATModel_dense1_single.h5')
inDir ='D:/WAT_HZmetadata/TPWS/ClusterBins_120dB_linear'
outDir = 'D:/WAT_HZmetadata/TPWS/ClusterBins_120dB_linear/labels'
for file in os.listdir(directory):
	fileName = os.fsdecode(file)
	if fileName.endswith("_toClassify.mat"): 
		f = h5py.File(fileName,'r')
		matData = f['nnVec'].value
		print(matData.shape)
		if matData.shape[0]>2:
			fileNameOut = fileName.replace('_toClassify.mat','predLab.mat')
			fileNameOut = os.path.join(outDir,fileNameOut)
			print(fileNameOut)
			matData = matData/np.std(matData,axis=0)
			predictedLabels = model.predict_classes(matData.transpose())
			probs = model.predict(matData.transpose())
			matOutData ={}
			matOutData[u'predLabels'] = predictedLabels+1
			matOutData[u'probs'] = probs
			hdf5storage.write(matOutData,'.',fileNameOut,matlab_compatible = True)
		continue
	else:
		continue
        
        