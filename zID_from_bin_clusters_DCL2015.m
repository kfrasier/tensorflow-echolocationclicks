toClassifyList = dir('J:\DCL\TPWS\toClassify\*__toClassify.mat');

labelDir = 'J:\DCL\TPWS\labels';
minCounts = 5% [10,10,5,5,10,5,10];% label-specific minimum counts required to consider labels, higher for dolphin than bw
countsPerBinAll = [];
runSum = 0;
binTimesAll = [];
falseIdx = [4,7];
idReducer = [1,2,3,NaN,4,5,NaN];
myTypeList = {'Gg';'HF';'LF';'Ship';'Zc';'Dolphin';'Echo'};
TPWSDir = 'J:\DCL\TPWS\';
%TPWSDir = 'D:\CANARC_PI_02\TPWS';
saveDir = TPWSDir;

for iFile = 1:length(toClassifyList)
    zID = [];
    % load cluster bins
    load(fullfile('J:\DCL\ClusterBins',...
        strrep(toClassifyList(iFile).name, '__toClassify',...
        '_clusterBins')));
    toClassify = load(fullfile(toClassifyList(iFile).folder,...
        toClassifyList(iFile).name))
    TPWSName = strrep(toClassifyList(iFile).name, '__toClassify','_TPWS');
    %[~,TPWSName] = fileparts(TPWSfilename);
    %binData(:).clustLabel = [];
    %binData.clustLabelScore = [];
    % load MTT
    load(fullfile(TPWSDir,[TPWSName]),'MTT', 'MPP')
    % prune low amp clicks
    %MTT = MTT(MPP>=120);
    MTT = MTT';
    % load labels
    labelName = strrep(TPWSName,'TPWS','predLab');
    zIDName = strrep(TPWSName,'TPWS','ID');
    zFDName = strrep(TPWSName,'TPWS','FD');

    load(fullfile(labelDir,labelName))
    probs = double(probs);
    countsPerBin = zeros(length(binData),21);
    tInt = vertcat(binData.tInt);
    for iC = 1:length(toClassify.catTimes)
        thisTimeBin = find(tInt(:,1) == toClassify.catTimes(iC));
        thisClickTimeSet = binData(thisTimeBin).clickTimes{toClassify.whichCell(iC)};
        % find the times of those clicks in the MTT vector
        % runSum = runSum+binData(iC).cInt;
        %for iS = 1:size(binData(iC).nSpec,2)
        thisAutoLabel = double(predLabels(iC));
            % Use the MTT idx to get the labels for those clicks
            %clustTimes = binData(iC).clickTimes{iS};
         [~,~,MTTIdx] = intersect(thisClickTimeSet,MTT);
         
         % output ID
         clickID = thisAutoLabel*ones(size(MTTIdx));
         zID = [zID;[MTT(MTTIdx),clickID]];
        
    end
    falseLabels = sum(bsxfun(@eq,zID(:,2),falseIdx),2)>0;
    zFD = zID(falseLabels,1);
    zID = zID(~falseLabels,:);
    zID(:,2) = idReducer(zID(:,2))';
    save(fullfile(saveDir,[zIDName]),'zID')
    save(fullfile(saveDir,[zFDName]),'zFD')
    countsPerBinAll = [countsPerBinAll;countsPerBin];
    tTemp = vertcat(binData.tInt);
    binTimesAll = [binTimesAll;tTemp(:,1)];
end
save(fullfile(labelDir,[zIDName,'_CpB.mat']),'countsPerBinAll','binTimesAll')