function labels = predict(bundle, raw)
[x,names]=har.features(raw,bundle.spec.featureSet);
assert(isequal(string(names(:)),string(bundle.featureNames(:))), 'Feature schema changed: retrain this model.');
z=(x-bundle.center)./bundle.scale;
if strcmp(bundle.spec.algorithm,'static_vote_svm')
    primary=string(predict(bundle.estimator.primary,z));votes=primary;
    for k=1:numel(bundle.estimator.specialists)
        specialist=bundle.estimator.specialists{k};
        assert(isequal(string(names(1:12)),string(specialist.featureNames)));
        votes(:,end+1)=string(predict(specialist.estimator,(x(:,1:12)-specialist.center)./specialist.scale)); %#ok<AGROW>
    end
    labels=har.staticVote(primary,votes);
elseif strcmp(bundle.spec.algorithm,'kmeans')
    [~,cluster]=min(pdist2(z,bundle.estimator,'squaredeuclidean'),[],2);
    labels=bundle.clusterLabels(cluster);
else
    labels=string(predict(bundle.estimator,z));
end
labels=labels(:);
assert(numel(labels)==size(raw,1) && all(ismember(labels,bundle.classes)), 'Invalid predictions.');
end
