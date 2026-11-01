% Export the saved model's actual decision evidence without training any model.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'matlab'));
d=har.loadData(root);saved=load(fullfile(root,'model/proto_v10/trained.mat'),'bundle');b=saved.bundle;
out=fullfile(root,'reports');if ~isfolder(out),mkdir(out);end
[x,names]=har.features(d.testRaw,b.spec.featureSet);
[primary,negLoss]=predict(b.estimator.primary,(x-b.center)./b.scale);primary=string(primary);
tree=har.predict(b.estimator.specialists{1},d.testRaw);
linear=har.predict(b.estimator.specialists{2},d.testRaw);final=har.predict(b,d.testRaw);
sorted=sort(negLoss,2,'descend');gap=sorted(:,1)-sorted(:,2);
trace=table((1:numel(primary))',d.testLabels,primary,tree,linear,final, ...
    ismember(primary,{'Sitting','Standing','Laying'}),primary~=final,gap, ...
    'VariableNames',{'row','truth','primarySVM','tree','linearSVM','final','voteApplied','voteChangedPrediction','primaryTopTwoNegLossGap'});
writetable(trace,fullfile(out,'classification-trace.csv'));
specialist=b.estimator.specialists{1};tr=specialist.estimator;nodes=find(tr.IsBranchNode);cuts=struct([]);
for k=1:numel(nodes)
    node=nodes(k);j=find(string(tr.PredictorNames)==string(tr.CutPredictor{node}));
    record=struct('node',node,'feature',specialist.featureNames(j),'standardizedCut',tr.CutPoint(node), ...
        'rawCut',tr.CutPoint(node)*specialist.scale(j)+specialist.center(j), ...
        'leftChild',tr.Children(node,1),'rightChild',tr.Children(node,2));
    if k==1,cuts=record;else,cuts(k)=record;end %#ok<AGROW>
end
writetable(struct2table(cuts),fullfile(out,'tree-branch-criteria.csv'));
focus=[8;10;105;155;176;201];
focusRecords=struct([]);
for k=1:numel(focus)
    i=focus(k);record=struct('row',i,'truth',d.testLabels(i),'primary',primary(i),'tree',tree(i),'linearSVM',linear(i),'final',final(i),'primaryNegativeLoss',negLoss(i,:));
    if k==1,focusRecords=record;else,focusRecords(k)=record;end %#ok<AGROW>
end
evidence=struct('model','proto_v10','modelSHA256',har.hashFile(fullfile(root,'model/proto_v10/trained.mat')), ...
    'featureNames',names,'primaryCenter',b.center,'primaryScale',b.scale, ...
    'classOrder',string(b.estimator.primary.ClassNames),'binaryLearners',numel(b.estimator.primary.BinaryLearners), ...
    'binaryLoss',b.estimator.primary.BinaryLoss,'parameters',b.parameters, ...
    'testWindows',numel(primary),'primaryStaticCandidates',sum(trace.voteApplied),'voteChangedWindows',sum(trace.voteChangedPrediction), ...
    'primaryCorrect',sum(primary==d.testLabels),'finalCorrect',sum(final==d.testLabels),'focus',focusRecords, ...
    'disclaimer','Labels guided prior development. Negative-loss values and gaps are not calibrated probabilities or causal feature attributions.');
% Primary-model sensitivity on the existing validation partition; no refitting.
p=har.protocol(root,d);validationX=har.features(d.trainRaw(p.validationRows,:,:),b.spec.featureSet);
z=(validationX-b.center)./b.scale;y=d.trainLabels(p.validationRows);
baselineAccuracy=mean(string(predict(b.estimator.primary,z))==y);
groups={1:6,7:12,13:16,17:20,21:26,27:32};
groupNames=["axis_means","axis_standard_deviations","acceleration_magnitude","gyroscope_magnitude","axis_correlations","band_power_fractions"];
sensitivity=struct([]);rng(51001,'twister');
for k=1:numel(groups)
    accuracies=zeros(5,1);
    for repeat=1:5
        perturbed=z;order=randperm(size(z,1));perturbed(:,groups{k})=z(order,groups{k});
        accuracies(repeat)=mean(string(predict(b.estimator.primary,perturbed))==y);
    end
    record=struct('group',groupNames(k),'firstColumn',groups{k}(1),'lastColumn',groups{k}(end), ...
        'repeats',5,'baselineAccuracy',baselineAccuracy,'meanPermutedAccuracy',mean(accuracies), ...
        'meanAccuracyDropPercentagePoints',100*(baselineAccuracy-mean(accuracies)), ...
        'stdAccuracyPercentagePoints',100*std(accuracies));
    if k==1,sensitivity=record;else,sensitivity(k)=record;end %#ok<AGROW>
end
sensitivityTable=sortrows(struct2table(sensitivity),'meanAccuracyDropPercentagePoints','descend');
writetable(sensitivityTable,fullfile(out,'primary-feature-group-sensitivity.csv'));
evidence.primaryValidationGroupSensitivity=table2struct(sensitivityTable);
har.writeJson(fullfile(out,'classification-evidence.json'),evidence);
assert(evidence.finalCorrect==201 && evidence.primaryCorrect==200 && evidence.voteChangedWindows==1);
assert(trace.row(trace.voteChangedPrediction)==155);
fprintf('DECISION_EVIDENCE_EXPORTED: %d features, %d pairwise learners, %d votes, %d changed prediction, %d/%d correct\n',numel(names),evidence.binaryLearners,evidence.primaryStaticCandidates,evidence.voteChangedWindows,evidence.finalCorrect,evidence.testWindows);
