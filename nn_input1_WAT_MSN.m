% Make a train set
% Load a TPWS
clearvars
trainFileList = [dir('I:\JAX13D_broad_metadata\TPWS_noMinPeakFr\ClusterBins_120dB_linear\JAX_D_13_disk04*_Delphin_timeSeries.mat');
    dir('I:\JAX13D_broad_metadata\TPWS_noMinPeakFr\ClusterBins_120dB_linear\JAX_D_13_disk02*_Delphin_timeSeries.mat')
    dir('I:\JAX13D_broad_metadata\TPWS_noMinPeakFr\ClusterBins_120dB_linear\JAX_D_13_disk01*_Delphin_timeSeries.mat')];
trainDataAll = [];
trainLabelsAll = [];
trainTimesAll = [];

for iFileTrain = 1:length(trainFileList)
    thisTPWSfile = fullfile(trainFileList(iFileTrain).folder,...
        trainFileList(iFileTrain).name);
    
    trainSet = load(thisTPWSfile,'trainLabel','trainMSN');
    trainSet.trainLabel(trainSet.trainLabel==8) = 1;
    trainSet.trainLabel(trainSet.trainLabel==3) = 2;
    trainSet.trainLabel(trainSet.trainLabel==14) = 2;
    trainSet.trainLabel(trainSet.trainLabel==9) = 6;
    trainSet.trainLabel(trainSet.trainLabel==11) = 10;
    trainSet.trainLabel(trainSet.trainLabel==12) = 10;
    trainSet.trainLabel(trainSet.trainLabel==13) = 10;

    % re-number
    trainSet.trainLabel(trainSet.trainLabel==10) = 3;

    
    
    if ~isempty(trainSet.trainLabel)
        trainSetPruned = expand_set_2(trainSet);
        if ~isempty(trainSetPruned.Label)
            meanTS = mean(trainSetPruned.MSN,2);
            stdTS = std(trainSetPruned.MSN-meanTS,0,2);
            trainDataAll = [trainDataAll;(trainSetPruned.MSN-meanTS)./stdTS];
            trainLabelsAll = [trainLabelsAll;trainSetPruned.Label];
        end
    end
    fprintf('Done with file %0.0f of %0.0f\n',iFileTrain,length(trainFileList))
    fprintf('%0.0f Training examples gathered\n',length(trainLabelsAll))
end
clear trainSet

% Make a train set
% Load a TPWS
testFileList = [dir('I:\JAX13D_broad_metadata\TPWS_noMinPeakFr\ClusterBins_120dB_linear\JAX_D_13_disk05*_Delphin_timeSeries.mat');
   dir('I:\JAX13D_broad_metadata\TPWS_noMinPeakFr\ClusterBins_120dB_linear\JAX_D_13_disk06*_Delphin_timeSeries.mat')];
testDataAll = [];
testLabelsAll = [];
testTimesAll = [];

for iFileTest = 1:length(testFileList)
    thisTPWSfile = fullfile(testFileList(iFileTest).folder,...
        testFileList(iFileTest).name);
    
    testSet = load(thisTPWSfile,'trainLabel','trainMSN');
    testSet.trainLabel(testSet.trainLabel==8) = 1;
    testSet.trainLabel(testSet.trainLabel==3) = 2;
    testSet.trainLabel(testSet.trainLabel==14) = 2;
    testSet.trainLabel(testSet.trainLabel==9) = 6;
    testSet.trainLabel(testSet.trainLabel==11) = 10;
    testSet.trainLabel(testSet.trainLabel==12) = 10;
    testSet.trainLabel(testSet.trainLabel==13) = 10;

    % re-number
    testSet.trainLabel(testSet.trainLabel==10) = 3;

    
    
    if ~isempty(testSet.trainLabel)
        testSetPruned = prune_set_2(testSet);
        if ~isempty(testSetPruned.Label)
            meanTS = mean(testSetPruned.MSN,2);
            stdTS = std(testSetPruned.MSN-meanTS,0,2);
            testDataAll = [testDataAll;(testSetPruned.MSN-meanTS)./stdTS];
            testLabelsAll = [testLabelsAll;testSetPruned.Label];
        end
    end
    fprintf('Done with file %0.0f of %0.0f\n',iFileTest,length(testFileList))
    fprintf('%0.0f Test examples gathered\n',length(testLabelsAll))
end

save('E:\Data\Papers\ClickClass2015\tensorflow\JAX13_unsupervisedTS_set.mat',...
    'testDataAll','testLabelsAll','trainDataAll','trainLabelsAll','-v7')
clear testSet


% to compare:
% confusionmat(double(testOut),y_test)