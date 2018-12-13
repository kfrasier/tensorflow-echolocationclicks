
fileList = dir('F:\HAT_A_06\HAT_A_06_d1-3_TPWS\labels\HAT_A_06_disk*_predLab.mat');
TPWSpath = 'F:\HAT_A_06\HAT_A_06_d1-3_TPWS';
for iFile =1:length(fileList)
    % load labels
    inFileName = fullfile(fileList(iFile).folder,fileList(iFile).name);
    load(inFileName)
    TPWSFilename = strrep(fileList(iFile).name,'predLab','TPWS1');
    load(fullfile(TPWSpath,TPWSFilename),'MTT')
    
    predLabels = double(predLabels)+1;
    probsMax = max(probs,[],2);
    
    %1 % mergeCells = {[]; %Gmsp
    %2 % [3,10]; %gervais
    %3 % []; %ud mid
    %4 % 5; %Pm
    %5 % 2; %3peak1
    %6 % []; %Gg
    %7 % []; %sowerby
    %8 % [];%kspp
    %9 % []; %echosndr
    %10 % [];% boat
    %11 % 4;%Zc
    %12 % 9}; %low noise
    
    zID = [MTT(probsMax>0.7),predLabels(probsMax>0.7)'];
    zID(:,2) = zID(:,2);% get rid of zero indexing
    falseIdxAll = [9,10,12];
    zFD =[];
    for iFalse = 1:length(falseIdxAll)
        falseSet = find(zID(:,2)==falseIdxAll(iFalse));
        zFD = [zFD;zID(falseSet,1)];
        zID(falseSet,:) = [];
    end
    
    % remap zID with numbers 1:12
    reMapId = [1, 1
        2,2
        3,3
        4,4
        5,5
        6,6
        7,7
        8,8
        11,9];
    
    for iID = 1:length(reMapId)
        thisId = find(zID(:,2)==reMapId(iID,1));
        zID(thisId,2) = reMapId(iID,2);
    end
    binSet5min = min(MTT):1/(24*12):max(MTT);
    [~,I] = histc(zID(:,1),binSet5min);
    uI = unique(I);
    for iBin = 1:length(uI)
        thisBin = uI(iBin);
        inThisBin = find(I==thisBin);
        [Ch,Ih] = histc(zID(inThisBin,2),1:13);
        for iT = 1:length(Ch)
            if Ch(iT)>0 && Ch(iT)<10
                zID(inThisBin(Ih == iT),2)= 0;
            end
        end
    end
    
    zID = zID(zID(:,2)>0,:);
    saveIDName = strrep(TPWSFilename,'TPWS','ID');
    saveFDName = strrep(TPWSFilename,'TPWS','FD');
    
    save(fullfile(TPWSpath,saveFDName),'zFD')
    save(fullfile(TPWSpath,saveIDName),'zID')
end