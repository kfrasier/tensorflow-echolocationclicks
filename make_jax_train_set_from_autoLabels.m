
clearvars
load('I:\JAX13D_broad_metadata\TPWS_noMinPeakFr\ClusterBins_120dB_linear\Composite90test\JAX_D_13_typesHR.mat',...
    'Tfinal','binTimes','nodeSet','subOrder')
TPWSList = dir('I:\JAX13D_broad_metadata\TPWS_noMinPeakFr\JAX_D_13_disk*TPWS1.mat');
binDir = 'I:\JAX13D_broad_metadata\TPWS_noMinPeakFr\ClusterBins_120dB_linear\';


for iTPWS = 31:50
    TPWSname = fullfile(TPWSList(iTPWS).folder,TPWSList(iTPWS).name);
    load(TPWSname,'MTT','MSN');
    disp(TPWSList(iTPWS).name)
    binFileName = strrep(TPWSList(iTPWS).name,'TPWS1.mat','clusters_PG0_PR90_MIN20_MOD0x_PPmin95FPincl.mat');
    load(fullfile(binDir,binFileName));
    disp(binFileName)
    
    clickNodeTimes = {};
    trainMSN = [];
    trainTimes = [];
    trainLabel = [];
    for iTF = 1:size(Tfinal,1)
        binTimesComposite = binTimes(nodeSet{iTF});
        binSubSet = subOrder(nodeSet{iTF});
        binTimesFile = vertcat(binData.tInt);
        [~,fileIntersect,binDataIntersect] = intersect(binTimesComposite,binTimesFile(:,1));
        binTimesCompIntersect = binTimesComposite(fileIntersect);
        binSubSetIntersect = binSubSet(fileIntersect);
        clickTimes = [];
        for iInt = 1:length(binTimesCompIntersect)
            
            clickTimes = [clickTimes;
                binData(binDataIntersect(iInt)).clickTimes{binSubSetIntersect(iInt)}];
        end
        clickNodeTimes{iTF} = clickTimes;
        
        [~,iTimes,~] = intersect(MTT,clickNodeTimes{iTF});
        trainTimes = [trainTimes;MTT(iTimes,:)];
        trainMSN = [trainMSN;MSN(iTimes,:)];
        trainLabel = [trainLabel;ones(size(iTimes))*iTF];
    end
    save(strrep(fullfile(binDir,binFileName),...
        'clusters_PG0_PR90_MIN20_MOD0x_PPmin95FPincl.mat',...
        'timeSeries.mat'),'trainMSN','trainLabel','clickNodeTimes');

end
