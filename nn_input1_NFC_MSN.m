% Make a train set
% Load a TPWS
clearvars
saveNameTrain = 'G:\NFC_A_02_ClusterBins_120dB\forNNet\NFC_HAT06A_train_1clust.mat';
saveNameTest = 'G:\NFC_A_02_ClusterBins_120dB\forNNet\NFC_HAT06A_test_1clust.mat';

trainFileList = [dir('G:\NFC_A_02_ClusterBins_120dB\forNNet\NFC_A_02_disk01*_Delphin_timeSeries.mat');
     dir('G:\NFC_A_02_ClusterBins_120dB\forNNet\NFC_A_02_disk04*_Delphin_timeSeries.mat')
     dir('G:\NFC_A_02_ClusterBins_120dB\forNNet\NFC_A_02_disk05*_Delphin_timeSeries.mat')
     dir('G:\NFC_A_02_ClusterBins_120dB\forNNet\NFC_A_02_disk07*_Delphin_timeSeries.mat')
     dir('F:\HAT_A_06\ClusterBins_120dB\forNNet\HAT_A_06_disk02*_Delphin_timeSeries.mat')
     dir('F:\HAT_A_06\ClusterBins_120dB\forNNet\HAT_A_06_disk04*_Delphin_timeSeries.mat')];
%      dir('F:\HAT_A_06\ClusterBins_120dB\forNNet\HAT_A_06_disk03*_Delphin_timeSeries.mat')];
trainDataAll = [];
trainLabelsAll = [];
trainTimesAll = [];

for iFileTrain = 1:length(trainFileList)
    thisTPWSfile = fullfile(trainFileList(iFileTrain).folder,...
        trainFileList(iFileTrain).name);
    
    trainSet = load(thisTPWSfile,'trainLabel','trainMSN');
    
    if ~isempty(trainSet.trainLabel)
        trainSetPruned = expand_set_2(trainSet);
        if ~isempty(trainSetPruned.Label)
            meanTS = mean(trainSetPruned.MSN,2);
            stdTS = std(trainSetPruned.MSN,0,2);
            trainDataAll = [trainDataAll;(trainSetPruned.MSN)./stdTS];
            trainLabelsAll = [trainLabelsAll;trainSetPruned.Label];
        end
    end
    fprintf('Done with file %0.0f of %0.0f\n',iFileTrain,length(trainFileList))
    fprintf('%0.0f Training examples gathered\n',length(trainLabelsAll))
end
clear trainSet
save(saveNameTrain,'trainDataAll','trainLabelsAll','-v7.3')

% Make a train set
% Load a TPWS
testFileList = [dir('G:\NFC_A_02_ClusterBins_120dB\forNNet\NFC_A_02_disk02*_Delphin_timeSeries.mat');
    dir('F:\HAT_A_06\ClusterBins_120dB\forNNet\NFC_A_02_disk03*_Delphin_timeSeries.mat')
     dir('F:\HAT_A_06\ClusterBins_120dB\forNNet\HAT_A_06_disk01*_Delphin_timeSeries.mat')
     dir('F:\HAT_A_06\ClusterBins_120dB\forNNet\HAT_A_06_disk05*_Delphin_timeSeries.mat')];

testDataAll = [];
testLabelsAll = [];
testTimesAll = [];

for iFileTest = 1:length(testFileList)
    thisTPWSfile = fullfile(testFileList(iFileTest).folder,...
        testFileList(iFileTest).name);
    
    testSet = load(thisTPWSfile,'trainLabel','trainMSN');    
    
    if ~isempty(testSet.trainLabel)
        testSetPruned = prune_set_2(testSet);
        if ~isempty(testSetPruned.Label)
            meanTS = mean(testSetPruned.MSN,2);
            stdTS = std(testSetPruned.MSN-meanTS,0,2);
            testDataAll = [testDataAll;(testSetPruned.MSN)./stdTS];
            testLabelsAll = [testLabelsAll;testSetPruned.Label];
        end
    end
    fprintf('Done with file %0.0f of %0.0f\n',iFileTest,length(testFileList))
    fprintf('%0.0f Test examples gathered\n',length(testLabelsAll))
end

save(saveNameTest,'testDataAll','testLabelsAll','-v7.3')
clear testSet


% to compare:
% confusionmat(double(testOut),y_test)