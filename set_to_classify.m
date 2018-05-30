load('K:\GOM_clickTypePaper_detections\TPWS\MC01_02_03_TPWS\Cluster_121\MC03_disk14_clusters_PG0_PR95_MIN100_MOD0_FPremov.mat')

sumSpecMat = cell2mat(sumSpec);
dTTMat = cell2mat(dTT);
dTTMat = dTTMat(:,1:41);
dTTMatNorm = dTTMat./max(dTTMat,[],2);
sumSpecMatNorm1 = sumSpecMat(:,1:121)-min(sumSpecMat(:,1:121),[],2);
sumSpecMatNorm = sumSpecMatNorm1./max(sumSpecMatNorm1,[],2);
toClassify = [sumSpecMatNorm,zeros(size(sumSpecMat,1),4),dTTMatNorm];

save('E:\Data\Papers\ClickClass2015\tensorflow\GOM_classify_MC.mat',...
    'toClassify','-v7')

nSpecMat = cell2mat(nSpec');

%%%%%%%
load('K:\GOM_clickTypePaper_detections\TPWS\GC01_02_03_TPWS\Cluster_prunebynode\GC03_disk14_clusters_PG0_PR95_MIN100_MOD0_FPremov.mat')

sumSpecMat = cell2mat(sumSpec);
dTTMat = cell2mat(dTT);
dTTMat = dTTMat(:,1:41);
dTTMatNorm = dTTMat./max(dTTMat,[],2);
sumSpecMatNorm1 = sumSpecMat(:,1:121)-min(sumSpecMat(:,1:121),[],2);
sumSpecMatNorm = sumSpecMatNorm1./max(sumSpecMatNorm1,[],2);
toClassify = [sumSpecMatNorm,zeros(size(sumSpecMat,1),4),dTTMatNorm];

save('E:\Data\Papers\ClickClass2015\tensorflow\GOM_classify_GC.mat',...
    'toClassify','-v7')
