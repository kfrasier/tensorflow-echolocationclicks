
clearvars
load('G:\NFC_A_02_ClusterBins_120dB\composite\NFC_A_02_recurs_1clust_typesHR.mat',...
    'Tfinal','tIntMat','prunedNodeSetCat','subOrder')
TPWSList = dir('G:\NFC_A_02_d1-3_TPWS\NFC_A_02_disk*TPWS1.mat');
binDir = 'G:\NFC_A_02_ClusterBins_120dB';
outDir = 'G:\NFC_A_02_ClusterBins_120dB\forNNet';

mergeCells = {[1,12]; %Gmsp
3; %gervais
4; %UD mid
[5,7]; %Pm
6; %3peak1
8; %Gg
[10,16]; % sowerby
13;%kspp
23; %echosndr
15}; %boat


prunedNodeSetCatM = [];
TfinalM = [];
for iM = 1:length(mergeCells)
    prunedNodeSetCatM{iM} = horzcat(prunedNodeSetCat{mergeCells{iM}});
    for iTF = 1:2
        TfinalM{iM,iTF} = vertcat(Tfinal{mergeCells{iM},iTF});
    end
end

for iTPWS = 1:length(TPWSList)
    TPWSname = fullfile(TPWSList(iTPWS).folder,TPWSList(iTPWS).name);
    load(TPWSname,'MTT','MSN');
    disp(TPWSList(iTPWS).name)
    binFileName = strrep(TPWSList(iTPWS).name,'TPWS1.mat','clusters_PG0_PR95_MIN25_MOD0x_PPmin95FPincl.mat');
    load(fullfile(binDir,binFileName));
    disp(binFileName)
    
    clickNodeTimes = {};
    trainMSN = [];
    trainTimes = [];
    trainLabel = [];
    for iTF = 1:size(TfinalM,1)
        binTimesComposite = tIntMat(prunedNodeSetCatM{iTF});
        binSubSet = subOrder(prunedNodeSetCatM{iTF});
        binTimesFile = vertcat(binData.tInt);
        [~,fileIntersect,binDataIntersect] = intersect(binTimesComposite,binTimesFile(:,1));
        binTimesCompIntersect = binTimesComposite(fileIntersect);
        binSubSetIntersect = binSubSet(fileIntersect);
        clickTimes = [];
        for iInt = 1:length(binTimesCompIntersect)
            if size(binData(binDataIntersect(iInt)).clickTimes)<binSubSetIntersect(iInt)
                warning('disagreement about subset field')
            else
                clickTimes = [clickTimes;
                    binData(binDataIntersect(iInt)).clickTimes{binSubSetIntersect(iInt)}];
            end
        end
        clickNodeTimes{iTF} = clickTimes;
        
        [~,iTimes,~] = intersect(MTT,clickNodeTimes{iTF});
        trainTimes = [trainTimes;MTT(iTimes,:)];
        trainMSN = [trainMSN;MSN(iTimes,:)];
        trainLabel = [trainLabel;ones(size(iTimes))*iTF];
    end
    save(strrep(fullfile(outDir,TPWSList(iTPWS).name),'TPWS1.mat',...
        'timeSeries.mat'),'trainMSN','trainLabel','clickNodeTimes');

end
