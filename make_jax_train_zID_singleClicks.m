load('I:\JAX13D_broad_metadata\TPWS_noMinPeakFr\labels\JAX_D_13_disk02a_DelphinpredLab_ClickLevel.mat')
load('I:\JAX13D_broad_metadata\TPWS_noMinPeakFr\JAX_D_13_disk02a_Delphin_TPWS1.mat','MTT')
predLabels = double(predLabels);
probsMax = max(probs,[],2);
zID = [MTT(probsMax>.8),predLabels(probsMax>.8)'];