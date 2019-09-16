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


directory = os.fsencode('J:/DCL/TPWS/toClassify')
# load trained network
os.chdir(directory)
model = load_model('J:/DCL/Composite_wICI/nnetInput/DCL2015_binLevel_Expand_network.h5')
inDir ='J:/DCL/TPWS/toClassify'
outDir = 'J:/DCL/TPWS/labels'
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
        
        