function cases = analyzeErrors(root)
% Explain baseline errors using scores, nearest fitting examples, and sensitivity.
data=har.loadData(root);saved=load(fullfile(root,'model/proto_v3/trained.mat'),'bundle');b=saved.bundle;
[x,names]=har.features(data.testRaw,b.spec.featureSet);
[fitX,~]=har.features(data.trainRaw(b.fitRows,:,:),b.spec.featureSet);
z=(x-b.center)./b.scale;fitZ=(fitX-b.center)./b.scale;
[pred,scores]=predict(b.estimator,z);pred=string(pred);classes=string(b.estimator.ClassNames);
errors=find(pred~=data.testLabels);out=fullfile(root,'evaluation','error_analysis');if ~isfolder(out),mkdir(out);end
cases=struct([]);
for k=1:numel(errors)
    row=errors(k);truth=data.testLabels(row);wrong=pred(row);
    actualClass=find(classes==truth);wrongClass=find(classes==wrong);
    trueRows=find(data.trainLabels(b.fitRows)==truth);wrongRows=find(data.trainLabels(b.fitRows)==wrong);
    trueMedian=median(fitX(trueRows,:),1);wrongMedian=median(fitX(wrongRows,:),1);
    [distance,nearest]=sort(pdist2(z(row,:),fitZ,'euclidean'));
    [~,nearTrue]=min(pdist2(z(row,:),fitZ(trueRows,:),'euclidean'));
    [~,nearWrong]=min(pdist2(z(row,:),fitZ(wrongRows,:),'euclidean'));
    trueRef=b.fitRows(trueRows(nearTrue));wrongRef=b.fitRows(wrongRows(nearWrong));
    gap=scores(row,actualClass)-scores(row,wrongClass);
    counterfactual=repmat(z(row,:),numel(names),1);
    for j=1:numel(names),counterfactual(j,j)=(trueMedian(j)-b.center(j))/b.scale(j);end
    [cfPred,cfScore]=predict(b.estimator,counterfactual);
    improvement=cfScore(:,actualClass)-cfScore(:,wrongClass)-gap;
    featureTable=table(names',x(row,:)',trueMedian',wrongMedian',((x(row,:)-trueMedian)./b.scale)', ...
        improvement,string(cfPred),'VariableNames',{'feature','observed','trueClassTrainingMedian','predictedClassTrainingMedian','deviationInTrainingSD','gapChangeIfSetToTrueMedian','counterfactualPrediction'});
    featureTable=sortrows(featureTable,'gapChangeIfSetToTrueMedian','descend');
    writetable(featureTable,fullfile(out,sprintf('row_%03d_features.csv',row)),'Encoding','UTF-8');
    count=min(10,numel(nearest));
    neighbors=table(b.fitRows(nearest(1:count)),data.trainLabels(b.fitRows(nearest(1:count))),distance(1:count)', ...
        'VariableNames',{'trainingRow','label','standardizedDistance'});
    writetable(neighbors,fullfile(out,sprintf('row_%03d_neighbors.csv',row)),'Encoding','UTF-8');
    c=struct('testRow',row,'truth',truth,'prediction',wrong,'classOrder',classes, ...
        'negativeLossScores',scores(row,:),'trueMinusPredictedScore',gap, ...
        'nearestTrueTrainingRow',trueRef,'nearestWrongTrainingRow',wrongRef, ...
        'topSensitiveFeatures',table2struct(featureTable(1:5,:)));
    if k==1,cases=c;else,cases(k)=c;end %#ok<AGROW>
    fig=figure('Visible','off','Position',[50 50 1200 800],'Color','white');
    layout=tiledlayout(3,2);t=(0:127)/50;
    for j=1:6
        nexttile;plot(t,squeeze(data.testRaw(row,:,j)),'k','LineWidth',1.3);hold on;
        plot(t,squeeze(data.trainRaw(trueRef,:,j)),'b');plot(t,squeeze(data.trainRaw(wrongRef,:,j)),'r:');
        title(strrep(char(names(j)),'_mean',''),'Interpreter','none');xlabel('Seconds');grid on;
    end
    legend('Misclassified test window','Nearest true-class training window','Nearest wrong-class training window','Location','best');
    title(layout,sprintf('Row %d: true %s, predicted %s',row,truth,wrong),'Interpreter','none');
    exportgraphics(fig,fullfile(out,sprintf('row_%03d_signals.png',row)),'Resolution',130);close(fig);
    fprintf('Row %d: %s -> %s; score gap %.4f; most sensitive: %s\n',row,truth,wrong,gap,featureTable.feature(1));
end
har.writeJson(fullfile(out,'baseline_errors.json'),struct('baseline','proto_v3','cases',cases, ...
    'interpretation','Negative-loss scores are not probabilities. Feature replacement is a sensitivity diagnostic, not a causal explanation or a physically realizable counterfactual. Nearest neighbors do not themselves determine the RBF SVM prediction.'));
end
