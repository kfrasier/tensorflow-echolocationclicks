function [trainSet] = prune_set_2(trainSet)

[labelCounts,~] = histc(trainSet.trainLabel,1:max(trainSet.trainLabel));
maxVal = min(min(labelCounts(labelCounts>0))*3,5000)% min(min(labelCounts(labelCounts>0))*2,10000);
trainSet.MSN = [];
trainSet.Label =[];
if maxVal>0
for labelNum = 1:max(trainSet.trainLabel)
    thisSet = find(trainSet.trainLabel==labelNum);
    numInSet = labelCounts(labelNum);
    newMSN = trainSet.trainMSN(thisSet(1:ceil(numInSet/maxVal):numInSet),:);
    trainSet.MSN = [trainSet.MSN; newMSN];
    trainSet.Label = [trainSet.Label;ones(size(newMSN,1),1)*labelNum];
end

end