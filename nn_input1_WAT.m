clearvars
%load('E:\Data\Papers\ClickClass2015\revamp\MC_GC_DT_MP_DC_autoCluster_95_1000ofEach_1-121_diff_min200_icifix_typesHR')
load('D:\WAT_HZmetadata\TPWS\ClusterBins_120dB_linear\singleCluster\WAT_HZ_01_typesHR.mat')
trainSet = {};    
testSet = {}; 

testLength = 30;
specTrain = [];
specTest = [];
iciTrain = [];
iciTest = [];
trainLabels = [];
testLabels = [];
repVal = [];

for iTF = 1:size(Tfinal,1)
    Tfinal{iTF,4} = Tfinal{iTF,4}';
end

for iTF = 1:size(Tfinal,2)
   Tfinal{3,iTF} = [Tfinal{3,iTF};Tfinal{7,iTF}];
   Tfinal{1,iTF} = [Tfinal{1,iTF};Tfinal{8,iTF};Tfinal{9,iTF}];
   Tfinal{5,iTF} = [Tfinal{5,iTF};Tfinal{11,iTF}];
end
Tfinal = Tfinal([1,2,3,4,5,10],:);

maxSubsetSize = max(cellfun(@length, Tfinal(:,1)));

trainLength = nan(size(Tfinal(:,1)));
for iTrain = 1:size(Tfinal,1)
    nSamples = size(Tfinal{iTrain,1},1);
    if size(Tfinal{iTrain,1},1)>testLength+5
        % if there are enough for testSet, add them
        trainLength(iTrain,1) = size(Tfinal{iTrain,1},1)-testLength;
    else
        % but sometimes there just aren't.
        trainLength(iTrain,1) = size(Tfinal{iTrain,1},1);
    end
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
    
    trainLabels = [trainLabels;ones(size(trainSet{iTrain,1}))'*(iTrain-1)];
    testLabels = [testLabels;ones(size(testSet{iTrain,1}))'*(iTrain-1)];
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

save('E:\Data\Papers\ClickClass2015\tensorflow\WAT_train_set1_singleCluster.mat',...
    'x_train','y_train','x_test','y_test','-v7')


% to compare:
% confusionmat(double(testOut),y_test)