
figure(5);colormap(jet)
for iPlot = 1:7
    subplot(3,3,iPlot)
    imagesc(sumSpecMat(predictedLabels==iPlot,1:121)');set(gca,'ydir','normal');
end
dTTMatNorm = dTTMat./max(dTTMat,[],2);
figure(6);colormap(jet)
for iPlot = 1:7
    subplot(3,3,iPlot)
    imagesc(dTTMatNorm(predictedLabels==iPlot,1:41)');set(gca,'ydir','normal');
end

probs = double(probs);
figure(7);clf;colormap(jet)
for iPlot = 1:7
    subplot(3,3,iPlot)
    imagesc(probs(predictedLabels==iPlot,:)');set(gca,'ydir','normal');
end
