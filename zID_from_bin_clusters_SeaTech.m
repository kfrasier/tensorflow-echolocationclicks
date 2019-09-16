%toClassifyList = dir('J:\Arctic_C2_10\Arctic_C2_10\toClassify\Arctic*__toClassify.mat');
toClassifyList = dir('J:\Arctic_C2_10\Arctic_C2_10\toClassify\CANARC*__toClassify.mat');

labelDir = 'J:\Arctic_C2_10\Arctic_C2_10\toClassify\labels';
minCounts = 5% [10,10,5,5,10,5,10];% label-specific minimum counts required to consider labels, higher for dolphin than bw
countsPerBinAll = [];
runSum = 0;
binTimesAll = [];
falseIdx = [5:8];
idReducer = [1,2,3,4,NaN,NaN,NaN,NaN];
myTypeList = {'Beluga'; 'BelugaBuzz';'Narwal';'NarwalLong';'Other1';'Other2';'Ship';'Sonar'};
TPWSDir = 'J:\Arctic_C2_10\Arctic_C2_10_TPWS';
%TPWSDir = 'D:\CANARC_PI_02\TPWS';
saveDir = TPWSDir;

for iFile = 2:length(toClassifyList)
    zID = [];
    % load cluster bins
    load(fullfile('J:\Arctic_C2_10\Arctic_C2_10\ClusterBins_120dB',...
        strrep(toClassifyList(iFile).name, '__toClassify',...
        '_Delphin_clusters_PR95_PPmin120')));
    toClassify = load(fullfile(toClassifyList(iFile).folder,...
        toClassifyList(iFile).name))
    TPWSName = strrep(toClassifyList(iFile).name, '__toClassify','_Delphin_TPWS1');
    %[~,TPWSName] = fileparts(TPWSfilename);
    %binData(:).clustLabel = [];
    %binData.clustLabelScore = [];
    % load MTT
    load(fullfile(TPWSDir,[TPWSName]),'MTT', 'MPP')
    % prune low amp clicks
    MTT = MTT(MPP>=120);
    
    % load labels
    labelName = strrep(TPWSName,'Delphin_TPWS1','predLab');
    zIDName = strrep(TPWSName,'Delphin_TPWS1','ID1');
    zFDName = strrep(TPWSName,'Delphin_TPWS1','FD1');

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