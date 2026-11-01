function result = metrics(truth, predicted, classes)
truth=string(truth(:)); predicted=string(predicted(:)); classes=string(classes(:));
assert(numel(truth)==numel(predicted) && ~isempty(truth), 'Invalid prediction count.');
assert(all(ismember(truth,classes)) && all(ismember(predicted,classes)), 'Unknown label.');
c=zeros(numel(classes));
for a=1:numel(classes)
    for b=1:numel(classes)
        c(a,b)=sum(truth==classes(a) & predicted==classes(b));
    end
end
tp=diag(c); support=sum(c,2); predictedCount=sum(c,1)';
precision=tp./max(predictedCount,1); recall=tp./max(support,1);
f1=2*precision.*recall./max(precision+recall,eps);
result=struct('accuracy',sum(tp)/sum(c(:)),'macroF1',mean(f1), ...
    'balancedAccuracy',mean(recall(support>0)),'correct',sum(tp),'total',sum(c(:)), ...
    'classes',classes,'confusion',c,'precision',precision,'recall',recall,'f1',f1,'support',support);
end
