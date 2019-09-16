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

directory = os.fsencode('J:/DCL/TPWS')
# load trained network
os.chdir(directory)
model = load_model('J:/DCL/forNNet_wICI/DCL_wICI_testOut.h5')
inDir ='J:/DCL/TPWS'
outDir = 'J:/DCL/TPWS/Labels'
for file in os.listdir(directory):
	fileName = os.fsdecode(file)
	if fileName.endswith("TPWS.mat"): 
		print(fileName)
		f = h5py.File(fileName,'r')
		MSN = f['MSN'].value
		MSP = f['MSP'].value
		print(MSN.shape)
		if MSN.shape[0]>2:
			fileNameOut = fileName.replace('TPWS.mat','predLab.mat')
			fileNameOut = os.path.join(outDir,fileNameOut)
			MSP = MSP[7:96,:]
			print(MSP.shape)
			
			
			MSP = MSP-np.min(MSP,axis=0)
			maxSpec = np.max(MSP,axis=0)
			print("maxSpec shape:")
			print(maxSpec.shape)
			normSpec = MSP/maxSpec[None,:]

			print(normSpec.shape)
			#print(np.max(MSN[94:105,0:100],axis=0))
			normTS = MSN/np.mean(np.max(MSN,axis = 0))
			#normTS = normTS[49:150,:]#maxTS = np.max(normTS,axis=0)
			matData = np.concatenate((normTS,normSpec))
			#print(matData[:,0])
			predictedLabels = model.predict_classes(matData.transpose())
			probs = model.predict(matData.transpose())
			matOutData ={}
			matOutData[u'predLabels'] = predictedLabels
			matOutData[u'probs'] = probs
			hdf5storage.write(matOutData,'.',fileNameOut,matlab_compatible = True)
			print('Done with file')
		continue
	else:
		continue
        
        