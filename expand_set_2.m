function trainSet = expand_set_2(trainSet)

[labelCounts,~] = histc(trainSet.trainLabel,1:max(trainSet.trainLabel));
trainSet.MSN = [];
trainSet.Label =[];
maxVal = min(max(labelCounts)*3,3000)% min(min(labelCounts(labelCounts>0))*2,10000);
for labelNum = 1:max(trainSet.trainLabel)
    if labelCounts(labelNum)>0

        thisSet = find(trainSet.trainLabel==labelNum);
        numInSet = labelCounts(labelNum);
        
        newMSN = trainSet.trainMSN(thisSet(round(1:numInSet/maxVal:numInSet)),:);
        trainSet.MSN = [trainSet.MSN; newMSN];
        trainSet.Label = [trainSet.Label;ones(size(newMSN,1),1)*labelNum];
        
    end
    histc(trainSet.Label,1:7)
end
