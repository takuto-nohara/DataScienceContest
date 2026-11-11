function e = explainWindow(bundle, raw)
% Explain one real inference; labels are deliberately not accepted as input.
assert(strcmp(bundle.spec.algorithm,'static_vote_svm'),'This explainer targets proto_v10.');
assert(size(raw,1)==1,'Select exactly one window.');
[x,names]=har.features(raw,bundle.spec.featureSet);
assert(isequal(string(names(:)),string(bundle.featureNames(:))));
z=(x-bundle.center)./bundle.scale;
[primary,negativeLoss]=predict(bundle.estimator.primary,z);primary=string(primary);
specialists=bundle.estimator.specialists;votes=primary;
for k=1:numel(specialists)
    votes(end+1)=har.predict(specialists{k},raw); %#ok<AGROW>
end
final=har.staticVote(primary,votes);
assert(final==har.predict(bundle,raw),'Explanation and production inference disagree.');
e=struct('raw',squeeze(raw),'features',x,'featureNames',names,'standardized',z, ...
    'center',bundle.center,'scale',bundle.scale,'primary',primary,'votes',votes, ...
    'final',final,'voteApplied',ismember(primary,["Sitting","Standing","Laying"]), ...
    'classes',string(bundle.estimator.primary.ClassNames),'negativeLoss',negativeLoss);
e.featureTable=table((1:numel(names))',names(:),x(:),bundle.center(:),bundle.scale(:),z(:), ...
    'VariableNames',{'column','feature','rawValue','trainingCenter','trainingScale','z'});
% Each plotted spectrum uses exactly the same bins as har.features/spectral32.
power=abs(fft(raw-mean(raw,2),[],2)).^2;
e.frequency=(1:64)*50/128;e.power=squeeze(power(:,2:65,:));
e.band=e.frequency>=0.5 & e.frequency<=3;
e.bandFraction=sum(e.power(e.band,:),1)./max(sum(e.power,1),eps);
assert(max(abs(e.bandFraction-x(27:32)))<1e-12);
% Scores and support-vector counts of the 15 saved pairwise SVMs.
svm=bundle.estimator.primary;pair=string.empty;positive=string.empty;negative=string.empty;
margin=[];selected=string.empty;support=[];
for k=1:numel(svm.BinaryLearners)
    learner=svm.BinaryLearners{k};
    pos=string(svm.ClassNames(svm.CodingMatrix(:,k)==1));
    neg=string(svm.ClassNames(svm.CodingMatrix(:,k)==-1));
    [label,score]=predict(learner,z);
    pair(end+1,1)=pos+" / "+neg;positive(end+1,1)=pos;negative(end+1,1)=neg; %#ok<AGROW>
    margin(end+1,1)=score(1,learner.ClassNames==1); %#ok<AGROW>
    if label==1,selected(end+1,1)=pos;else,selected(end+1,1)=neg;end %#ok<AGROW>
    support(end+1,1)=size(learner.SupportVectors,1); %#ok<AGROW>
end
e.pairTable=table(pair,positive,negative,margin,selected,support);
% Follow the numeric cuts of the saved supplementary tree, not a surrogate.
specialist=specialists{1};tree=specialist.estimator;
treeZ=(x(1:12)-specialist.center)./specialist.scale;
node=1;nodes=[];features=string.empty;values=[];cuts=[];branches=string.empty;
while tree.IsBranchNode(node)
    j=find(string(tree.PredictorNames)==string(tree.CutPredictor{node}));
    nodes(end+1,1)=node;features(end+1,1)=specialist.featureNames(j); %#ok<AGROW>
    values(end+1,1)=x(j);cuts(end+1,1)=tree.CutPoint(node)*specialist.scale(j)+specialist.center(j); %#ok<AGROW>
    if treeZ(j)<tree.CutPoint(node),side=1;branches(end+1,1)="< : left";else,side=2;branches(end+1,1)=">= : right";end %#ok<AGROW>
    node=tree.Children(node,side);
end
e.treeLeaf=node;e.treeLabel=string(tree.NodeClass{node});
assert(e.treeLabel==votes(2),'Tree path does not reproduce saved tree prediction.');
e.treeTable=table(nodes,features,values,cuts,branches,'VariableNames',{'node','feature','rawValue','rawCut','branch'});
end
