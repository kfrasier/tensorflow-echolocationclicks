load('D:\WAT_HZmetadata\TPWS\ClusterBins_120dB_linear\labels\WAT_HZ_01_disk05_predLab.mat')
load('D:\WAT_HZmetadata\TPWS\ClusterBins_120dB_linear\WAT_HZ_01_disk05_Delphin_clusters_PG0_PR95_MIN100_MOD0x_PPmin95FPincl.mat')
load('D:\WAT_HZmetadata\TPWS\ClusterBins_120dB_linear\WAT_HZ_01_disk05__toClassify.mat')

probs = double(probs);
predLabels = double(predLabels);
allTimes = vertcat(binData.tInt);
zID =[];
for iTime = 1:length(catTimes)
    if  probs(iTime,predLabels(iTime))>=.99
        thisTime = catTimes(iTime);
        timeIdx = find(allTimes(:,1) == thisTime);
    
        subIdx = whichCell(iTime);
    
        newIdTime = binData(timeIdx).clickTimes{subIdx};
        newIdLabel = repmat(predLabels(iTime),size(newIdTime));
        zID = [zID;[newIdTime,newIdLabel]];
    end
end
