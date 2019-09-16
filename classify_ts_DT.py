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


# directory = os.fsencode('I:/JAX13D_broad_metadata/TPWS_noMinPeakFr')
inDir ='D:/WAT_BS_01_Detector/TPWS'
directory = os.fsencode(inDir)
# load trained network
os.chdir(directory)
model = load_model('E:/Data/CIMAGEIII/GOM_no700_120dB_unsup.h5')
outDir = 'D:/WAT_BS_01_Detector/TPWS/labels'
for file in os.listdir(directory):
	fileName = os.fsdecode(file)
	if fileName.endswith("TPWS1.mat"): 
		f = h5py.File(fileName,'r')
		matData = f['MSN'].value
		tDiff = np.diff(f['MTT'].value)*24*60*60
		tDiff[tDiff>1] = 1
		print(tDiff.shape)
		tDiff = np.concatenate((tDiff,0))
		print(tDiff.shape)
		print(matData.shape)
		matData = np.concatenate((matData,tDiff),axis==0)
		print(matData.shape)
		if matData.shape[0]>2:
			fileNameOut = fileName.replace('TPWS1.mat','predLab.mat')
			fileNameOut = os.path.join(outDir,fileNameOut)
			print(fileNameOut)
			matData = matData/np.std(matData,axis=0)
			predictedLabels = model.predict_classes(matData.transpose())
			probs = model.predict(matData.transpose())
			matOutData ={}
			matOutData[u'predLabels'] = predictedLabels
			matOutData[u'probs'] = probs
			hdf5storage.write(matOutData,'.',fileNameOut,matlab_compatible = True)
		continue
	else:
		continue
