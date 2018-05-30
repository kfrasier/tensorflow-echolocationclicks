%load('E:\Data\Papers\ClickClass2015\revamp\MC_GC_DT_MP_DC_autoCluster_95_1000ofEach_1-121_diff_min200_icifix_typesHR')
load('K:\GOM_clickTypePaper_detections\TPWS\MC_GC_DTmerge\MC_GC_DT_MP_DC_autoCluster_95_2000ofEach_1-121_diff_typesHR.mat')
trainSet = {};    
testSet = {}; 

testLength = 50;
specTrain = [];
specTest = [];
iciTrain = [];
iciTest = [];
trainLabels = [];
testLabels = [];
repVal = [];

maxSubsetSize = max(cellfun(@length, Tfinal(:,1)));

trainLength = nan(size(Tfinal(:,1)));
for iTrain = 1:size(Tfinal,1)
    nSamples = size(Tfinal{iTrain,1},1);
    trainLength(iTrain,1) = size(Tfinal{iTrain,1},1)-25;
    trainTemp = randperm(nSamples,trainLength(iTrain,1));
    testTemp = setdiff(1:nSamples,trainTemp);
    
    if size(trainTemp,2)<maxSubsetSize
        repVal(iTrain,1) = floor(maxSubsetSize/size(trainTemp,2));
        trainTemp = repmat(trainTemp,1,repVal(iTrain,1));
    end
    
    trainSet{iTrain,1} = trainTemp;
    testSet{iTrain,1} = testTemp;
    
    specTrain = [specTrain;Tfinal{iTrain,1}(trainSet{iTrain,1},1:121)];
    specTest = [specTest;Tfinal{iTrain,1}(testSet{iTrain,1},1:121)];
    
    iciTrain = [iciTrain;Tfinal{iTrain,2}(trainSet{iTrain,1},:)];
    iciTest = [iciTest;Tfinal{iTrain,2}(testSet{iTrain,1},:)];
    
    trainLabels = [trainLabels;ones(size(trainSet{iTrain,1}))'*iTrain];
    testLabels = [testLabels;ones(size(testSet{iTrain,1}))'*iTrain];
end
specTrainNorm1 = specTrain-min(specTrain,[],2);
specTrainNorm = specTrainNorm1./max(specTrainNorm1,[],2);

specTestNorm1 = specTest-min(specTest,[],2);
specTestNorm = specTestNorm1./max(specTestNorm1,[],2);

iciTrainNorm = iciTrain./max(iciTrain,[],2);
iciTestNorm = iciTest./max(iciTest,[],2);

x_train = [specTrainNorm,zeros(size(specTrainNorm,1),4),iciTrainNorm];
y_train = trainLabels;
x_test = [specTestNorm,zeros(size(specTestNorm,1),4),iciTestNorm];
y_test = testLabels;

save('E:\Data\Papers\ClickClass2015\tensorflow\GOM_train_set_2000.mat',...
    'x_train','y_train','x_test','y_test','-v7')


% to compare:
% confusionmat(double(testOut),y_test)