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


directory = os.fsencode('J:/CalCOFI_Detections_All/CalCOFI_TPWS')
# load trained network
os.chdir(directory)
model = load_model('J:/CalCOFI_Detections_All/CalCOFI_TPWS/mergeSRates_cluster_bins_95/forNNet/CC_noICI.h5')
inDir ='J:/CalCOFI_Detections_All/CalCOFI_TPWS/'
outDir = 'J:/CalCOFI_Detections_All/CalCOFI_TPWS/labels'
for file in os.listdir(directory):
	fileName = os.fsdecode(file)
	#print(fileName)
	if fileName.endswith("TPWS3.mat"): 
		f = h5py.File(fileName,'r')
		MSN = f['MSN'].value
		MSP = f['MSP'].value
		print(MSN.shape)
		if MSN.shape[0]>2:
			fileNameOut = fileName.replace('TPWS3.mat','predLab.mat')
			fileNameOut = os.path.join(outDir,fileNameOut)
			#print(fileNameOut)
			#MSN = MSN[74:500,:]
			print(MSN.shape)
			normTS = MSN/np.max(np.abs(MSN),axis=0)
			# maxTS = np.max(normTS,axis=0)
			
			specMinNorm = MSP-np.min(MSP,axis=0)
			normSpec = specMinNorm/np.max(specMinNorm,axis=0)
			matData = np.concatenate((normTS,normSpec))
			print(matData.shape)
			predictedLabels = model.predict_classes(matData.transpose())
			probs = model.predict(matData.transpose())
			matOutData ={}
			matOutData[u'predLabels'] = predictedLabels
			matOutData[u'probs'] = probs
			hdf5storage.write(matOutData,'.',fileNameOut,matlab_compatible = True)
		continue
	else:
		continue
        
        