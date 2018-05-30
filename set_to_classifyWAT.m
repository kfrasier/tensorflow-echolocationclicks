clearvars;
fList = dir('D:\WAT_HZmetadata\TPWS\ClusterBins_120dB_linear\*incl.mat');

for iFile = 1:length(fList)
    load(fullfile(fList(iFile).folder,fList(iFile).name));
    catTimes = [];
    whichCell = [];
    for iTimes = 1:size(binData,1)
        catTimes = [catTimes;repmat(binData(iTimes).tInt(1),...
            size(binData(iTimes).nSpec,2),1)];
        whichCell = [whichCell;[1:size(binData(iTimes).nSpec,2)]'];
    end
    catNSpec = horzcat(binData.nSpec)';
    catSpec = vertcat(binData.sumSpec);  
    catSpecMin = catSpec - repmat(min(catSpec,[],2),1,size(catSpec,2));
    catSpecNorm = catSpecMin./max(catSpecMin,[],2);

    catDTT = vertcat(binData.dTT);
    nnVec = [catSpecNorm,...
        zeros(size(catTimes,1),4),...
        catDTT./max(catDTT,[],2)];
    nnVec(catNSpec<100,:) = [];
    catTimes(catNSpec<100,:) = [];
    whichCell(catNSpec<100,:) = [];
    outFileName = strrep(fList(iFile).name,...
        'Delphin_clusters_PG0_PR95_MIN100_MOD0x_PPmin95FPincl.mat',...
        '_toClassify.mat');
    save(fullfile(fList(iFile).folder,outFileName),'nnVec','catTimes','whichCell')
end
