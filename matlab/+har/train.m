function train(root, ids)
if nargin<2, ids=string.empty; end
data=har.loadData(root); protocol=har.protocol(root,data);
folders=dir(fullfile(root,'model','proto_*')); folders=folders([folders.isdir]);
assert(~isempty(folders),'No prototype specifications found.');
if ~isempty(ids),assert(all(ismember(string(ids),string({folders.name}))),'Unknown requested model ID.');end
for k=1:numel(folders)
    folder=fullfile(folders(k).folder,folders(k).name);
    spec=jsondecode(fileread(fullfile(folder,'spec.json')));
    if ~isempty(ids) && ~ismember(string(spec.id),string(ids)), continue; end
    assert(strcmp(spec.id,folders(k).name) && strcmp(spec.stage,'prototype'), 'Invalid prototype ID/stage.');
    assert(spec.trainFraction>0 && spec.trainFraction<=1, 'Invalid trainFraction.');
    rng(spec.seed,'twister');
    rows=protocol.fitRows(:);
    if spec.trainFraction<1
        selected=[];
        for label=data.classes'
            candidates=rows(data.trainLabels(rows)==label);
            selected=[selected;candidates(randperm(numel(candidates),max(1,floor(numel(candidates)*spec.trainFraction))))]; %#ok<AGROW>
        end
        rows=sort(selected);
    end
    [x,names]=har.features(data.trainRaw(rows,:,:),spec.featureSet);
    y=data.trainLabels(rows);
    center=mean(x,1); scale=std(x,0,1); scale(scale<1e-12)=1;
    z=(x-center)./scale;
    fprintf('Training %s (%s, %d features, %d rows)\n',spec.id,spec.algorithm,size(x,2),numel(rows));
    timer=tic;
    clusterLabels=string.empty;
    switch spec.algorithm
        case 'tree'
            estimator=compact(fitctree(z,y,'MaxNumSplits',100,'MinLeafSize',5));
            parameters=struct('MaxNumSplits',100,'MinLeafSize',5);
        case {'linear_svm','rbf_svm'}
            kernel='linear'; if strcmp(spec.algorithm,'rbf_svm'), kernel='gaussian'; end
            learner=templateSVM('KernelFunction',kernel,'KernelScale','auto','BoxConstraint',1,'Standardize',false);
            estimator=compact(fitcecoc(z,y,'Learners',learner,'Coding','onevsone'));
            parameters=struct('kernel',kernel,'KernelScale','auto','BoxConstraint',1,'Coding','onevsone');
        case 'bagged_trees'
            learner=templateTree('MinLeafSize',3,'MaxNumSplits',200);
            estimator=compact(fitcensemble(z,y,'Method','Bag','NumLearningCycles',100,'Learners',learner));
            parameters=struct('NumLearningCycles',100,'MinLeafSize',3,'MaxNumSplits',200);
        case 'knn'
            estimator=fitcknn(z,y,'NumNeighbors',5,'DistanceWeight','squaredinverse','Standardize',false);
            parameters=struct('NumNeighbors',5,'DistanceWeight','squaredinverse');
        case 'kmeans'
            [assignments,estimator]=kmeans(z,24,'Replicates',5,'MaxIter',500,'EmptyAction','singleton');
            clusterLabels=strings(24,1);
            for j=1:24
                counts=arrayfun(@(label)sum(y(assignments==j)==label),data.classes);
                [~,winner]=max(counts); clusterLabels(j)=data.classes(winner);
            end
            parameters=struct('clusters',24,'Replicates',5,'MaxIter',500,'mapping','training majority class per cluster');
        otherwise
            error('Unknown algorithm: %s',spec.algorithm);
    end
    duration=toc(timer);
    bundle=struct('schemaVersion',1,'spec',spec,'parameters',parameters,'estimator',estimator, ...
        'center',center,'scale',scale,'featureNames',names,'clusterLabels',clusterLabels, ...
        'classes',data.classes,'fitRows',rows,'protocol',protocol,'trainingSeconds',duration, ...
        'matlabVersion',version,'createdAt',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ssXXX')));
    save(fullfile(folder,'trained.mat'),'bundle','-v7');
    metadata=rmfield(bundle,{'estimator','center','scale','clusterLabels','protocol'});
    metadata.fingerprint=data.fingerprint;
    metadata.protocolVersion=protocol.version;
    metadata.artifactSHA256=har.hashFile(fullfile(folder,'trained.mat'));
    har.writeJson(fullfile(folder,'manifest.json'),metadata);
    fprintf('Saved %s in %.2f seconds\n',spec.id,duration);
end
end
