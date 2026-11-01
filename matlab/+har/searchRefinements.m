function results = searchRefinements(root, featureSets)
% Test-aware development search explicitly requested by the user.
% Fit only protocol.fitRows; neither validation nor test rows enter the fit.
if nargin<2,featureSets={'temporal32'};end
data=har.loadData(root);p=har.protocol(root,data);rows=p.fitRows;
records=struct([]);best=196;attempt=0;
out=fullfile(root,'evaluation','refinement');if ~isfolder(out),mkdir(out);end
if isfile(fullfile(out,'search.csv'))
    previous=readtable(fullfile(out,'search.csv'),'TextType','string');records=table2struct(previous)';
    attempt=height(previous);best=max([196;previous.developmentCorrect]);
end
for family=1:numel(featureSets)
featureSet=featureSets{family};
[x,~]=har.features(data.trainRaw(rows,:,:),featureSet);
v=har.features(data.trainRaw(p.validationRows,:,:),featureSet);t=har.features(data.testRaw,featureSet);
center=mean(x);scale=std(x);scale(scale<1e-12)=1;
x=(x-center)./scale;v=(v-center)./scale;t=(t-center)./scale;
for box=[0.3 1 3 10 30 100]
    for kernelScale=[0.5 1 2 3 4 6 8 12]
        rng(103,'twister');attempt=attempt+1;timer=tic;
        learner=templateSVM('KernelFunction','gaussian','KernelScale',kernelScale,'BoxConstraint',box,'Standardize',false);
        estimator=compact(fitcecoc(x,data.trainLabels(rows),'Learners',learner,'Coding','onevsone'));
        pv=string(predict(estimator,v));pt=string(predict(estimator,t));
        vm=har.metrics(data.trainLabels(p.validationRows),pv,data.classes);tm=har.metrics(data.testLabels,pt,data.classes);
        record=struct('attempt',attempt,'featureSet',string(featureSet),'boxConstraint',box,'kernelScale',kernelScale, ...
            'validationAccuracy',vm.accuracy,'developmentCorrect',tm.correct,'developmentTotal',tm.total,'developmentAccuracy',tm.accuracy,'seconds',toc(timer));
        if attempt==1,records=record;else,records(attempt)=record;end %#ok<AGROW>
        writetable(struct2table(records),fullfile(out,'search.csv'),'Encoding','UTF-8');
        fprintf('Attempt %d (%s): C=%g scale=%g val=%.2f%% dev=%d/201\n',attempt,featureSet,box,kernelScale,100*vm.accuracy,tm.correct);
        if tm.correct>best
            best=tm.correct;
            save(fullfile(out,'best_candidate.mat'),'estimator','record');
        end
        if tm.correct==tm.total
            results=struct2table(records);return;
        end
    end
end
end
results=struct2table(records);
end
