binClustFList = dir('D:\WAT_BS_01_Detector\ClusterBins_120dB\WAT_BS_01*.mat');

labelDir = 'D:\WAT_BS_01_Detector\TPWS\labels';
TPWSDir = 'D:\WAT_BS_01_Detector\TPWS';
saveDir = TPWSDir;
minCounts = 10;% [10,10,5,5,10,5,10];% label-specific minimum counts required to consider labels, higher for dolphin than bw
countsPerBinAll = [];
runSum = 0;
binTimesAll = [];
falseIdx = [2,12,15];
idReducer = [1,NaN,2:10,NaN,11:12, NaN, 13:16];
myTypeList = {'blainvilles'; 'boats';'CT2';'CT3';'CT4';'CT5';'CT7';'CT8';'CT9';'CT10';
    'cuviers';'echosounder';'gervais';'kogia';'noise';'rissos';'sowerbys';'sperm';'trues'};


for iFile = 1:length(binClustFList)
    zID = [];
    % load cluster bins
    load(fullfile(binClustFList(iFile).folder,binClustFList(iFile).name))
    [~,TPWSName] = fileparts(TPWSfilename);
    %binData(:).clustLabel = [];
    %binData.clustLabelScore = [];
    % load MTT
    load(fullfile(TPWSDir,[TPWSName,'.mat']),'MTT', 'MPP')
    % prune low amp clicks
    MTT = MTT(MPP>=120);
    
    % load labels
    labelName = strrep(TPWSName,'TPWS1','predLab');
    zIDName = strrep(TPWSName,'TPWS1','ID1');
    zFDName = strrep(TPWSName,'TPWS1','FD1');

    load(fullfile(labelDir,[labelName,'.mat']))
    probs = double(probs);
    countsPerBin = zeros(length(binData),21);
    for iC = 1:length(binData)
        % find the times of those clicks in the MTT vector
        % runSum = runSum+binData(iC).cInt;
        %for iS = 1:size(binData(iC).nSpec,2)

            % Use the MTT idx to get the labels for those clicks
            %clustTimes = binData(iC).clickTimes{iS};
            %[~,~,MTTIdx] = intersect(clustTimes,MTT);
            MTTIdx = find(MTT>=binData(iC).tInt(1,1)&MTT<binData(iC).tInt(1,2));
            
            % Figure out what the most common label is
            labelSet = double(predLabels(MTTIdx)+1);
            probSet = probs(MTTIdx,:);
            probIdx = sub2ind(size(probSet),[1:size(MTTIdx)]',labelSet');
            labelProb = probSet(probIdx);
            labelProbStrong = labelProb;
            labelProbStrong(labelProbStrong<.5) = 0;
            uLabels = unique(labelSet);
            minCountsSet = minCounts;% minCounts(uLabels);
            pClustLabel1=[];
            pClustLabel2 = [];
            nInSet1 = [];
            nInSet = [];
            for iProb = 1:length(uLabels)
                inSet = (labelSet==uLabels(iProb));
                % pClustLabelMed(iProb) = median(labelProb(inSet));
                %myProb = labelProbStrong(inSet);
                %pClustLabel1(iProb) = mean(myProb(myProb>0));
                pClustLabel2(iProb) = sum(labelProb(inSet))/sum(inSet);
                %nInSet1(iProb) = sum(myProb>0);
                nInSet(iProb) = sum(inSet);
            end
            if size(pClustLabel2,2)>1
                %pClustLabel1(nInSet1<minCountsSet) = 0;
                pClustLabel2(nInSet<minCountsSet) = 0;
            end
            if max(pClustLabel2)<.4 % no strong labels
               countsPerBin(iC,21) = countsPerBin(iC,21)+size(MTTIdx,1);
            elseif sum(pClustLabel2>=.4)==1
               % only one strong label in this set, assign everything to
               % that label.
               [bestScore,bestLabelIdx] = max(pClustLabel2);
               iDnum = uLabels(bestLabelIdx);
               countsPerBin(iC,iDnum) = countsPerBin(iC,iDnum)+...
                    size(MTTIdx,1);
               zID = [zID;[MTT(MTTIdx),repmat(iDnum,size(MTTIdx,1),...
                    size(MTTIdx,2))]];
            elseif sum(pClustLabel2>=.4)> 1 
                % there are multiple options
                possibleLabelIdx = find(pClustLabel2>=.4);
                possibleLabels = uLabels(pClustLabel2>=.4);
                prunedProbs = probSet(:,possibleLabels);
                [C,I] = max(prunedProbs,[],2);
                clickID = possibleLabels(I)';

                % output ID
                zID = [zID;[MTT(MTTIdx),clickID]];
                %myPerc =  binData(iC).percSpec(iS)./sum(binData(iC).percSpec);
                for uI = 1:length(possibleLabels)
                    iDnum = possibleLabels(uI);
                    countsPerBin(iC,iDnum) = countsPerBin(iC,iDnum)+...
                            sum(clickID==iDnum);
                end
            end
        % end
    end
    falseLabels = sum(bsxfun(@eq,zID(:,2),falseIdx),2)>0;
    zFD = zID(falseLabels,1);
    zID = zID(~falseLabels,:);
    zID(:,2) = idReducer(zID(:,2))';
    save(fullfile(saveDir,[zIDName,'.mat']),'zID')
    save(fullfile(saveDir,[zFDName,'.mat']),'zFD')
    countsPerBinAll = [countsPerBinAll;countsPerBin];
    tTemp = vertcat(binData.tInt);
    binTimesAll = [binTimesAll;tTemp(:,1)];
end
save(fullfile(labelDir,[zIDName,'_CpB.mat']),'countsPerBinAll','binTimesAll')