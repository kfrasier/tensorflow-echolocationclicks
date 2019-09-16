clearvars;
fList = dir('J:\DCL\ClusterBins\*Bins.mat');
outDir = 'J:\DCL\TPWS\toClassify';
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
    %catSpecMin = catSpec - repmat(min(catSpec,[],2),1,size(catSpec,2));
    %catSpecNorm = catSpecMin./max(catSpecMin,[],2);

    catDTT = vertcat(binData.dTT);
    nnVec = [catSpec,zeros(size(catSpec,1),1),...
        catDTT./max(catDTT,[],2)]; %zeros(size(catTimes,1),4),...
%     nnVec(catNSpec<50,:) = []; 
%     catTimes(catNSpec<50,:) = [];
%     whichCell(catNSpec<50,:) = [];
    outFileName = strrep(fList(iFile).name,...
        'clusterBins.mat',...
        '_toClassify.mat');
    save(fullfile(outDir,outFileName),'nnVec',...
        'catTimes','whichCell')
end
