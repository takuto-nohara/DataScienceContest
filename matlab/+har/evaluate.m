function leaderboard = evaluate(root, ids, outputName)
% Discover any number of saved prototypes. Never train, tune, or refit here.
if nargin<2, ids=string.empty; end
if nargin<3, outputName='latest'; end
assert(~isempty(regexp(outputName,'^[A-Za-z0-9_-]+$','once')), 'Invalid output name.');
data=har.loadData(root); protocol=har.protocol(root,data);
folders=dir(fullfile(root,'model','proto_*')); folders=folders([folders.isdir]);
if ~isempty(ids)
    assert(all(ismember(string(ids),string({folders.name}))), 'Unknown requested model ID.');
    folders=folders(ismember(string({folders.name}),string(ids)));
end
assert(~isempty(folders), 'No prototypes found.');
out=fullfile(root,'evaluation',outputName); if ~isfolder(out),mkdir(out);end
records=struct([]); details=struct([]);
for k=1:numel(folders)
    folder=fullfile(folders(k).folder,folders(k).name);
    file=fullfile(folder,'trained.mat'); assert(isfile(file),'Missing model: %s',file);
    metadata=jsondecode(fileread(fullfile(folder,'manifest.json')));
    assert(strcmp(metadata.artifactSHA256,har.hashFile(file)), 'Model artifact checksum mismatch.');
    saved=load(file,'bundle'); bundle=saved.bundle;
    currentSpec=jsondecode(fileread(fullfile(folder,'spec.json')));
    assert(isequal(currentSpec,bundle.spec), 'Spec changed: retrain %s',folders(k).name);
    assert(isequal(bundle.protocol.fingerprint,data.fingerprint), 'Model/data fingerprint mismatch.');
    assert(isequal(bundle.protocol.fitRows(:),protocol.fitRows(:)) && isequal(bundle.protocol.validationRows(:),protocol.validationRows(:)), 'Models use different protocols.');
    assert(all(ismember(bundle.fitRows,protocol.fitRows)) && isempty(intersect(bundle.fitRows,protocol.validationRows)), 'Training contamination.');
    assert(isequal(string(bundle.classes(:)),data.classes), 'Class order mismatch.');
    timer=tic;
    validation=har.predict(bundle,data.trainRaw(protocol.validationRows,:,:));
    validationSeconds=toc(timer);
    timer=tic; predicted=har.predict(bundle,data.testRaw); testSeconds=toc(timer);
    v=har.metrics(data.trainLabels(protocol.validationRows),validation,data.classes);
    t=har.metrics(data.testLabels,predicted,data.classes);
    info=dir(file);
    record=struct('model',string(bundle.spec.id),'algorithm',string(bundle.spec.algorithm), ...
        'featureSet',string(bundle.spec.featureSet),'features',numel(bundle.featureNames), ...
        'trainingRows',numel(bundle.fitRows),'validationAccuracy',v.accuracy,'validationMacroF1',v.macroF1, ...
        'testAccuracy',t.accuracy,'testMacroF1',t.macroF1,'testBalancedAccuracy',t.balancedAccuracy, ...
        'testCorrect',t.correct,'testTotal',t.total,'trainingSeconds',bundle.trainingSeconds, ...
        'validationSeconds',validationSeconds,'testSeconds',testSeconds,'modelBytes',info.bytes);
    detail=struct('model',bundle.spec.id,'validation',v,'test',t);
    if k==1
        records=record; details=detail;
    else
        records(k)=record; details(k)=detail; %#ok<AGROW>
    end
    writetable(table((1:numel(predicted))',data.testLabels,predicted,'VariableNames',{'row','truth','prediction'}),fullfile(out,[bundle.spec.id '_predictions.csv']));
    writematrix(t.confusion,fullfile(out,[bundle.spec.id '_confusion.csv']));
    writetable(table(data.classes,t.precision,t.recall,t.f1,t.support,'VariableNames',{'class','precision','recall','f1','support'}),fullfile(out,[bundle.spec.id '_per_class.csv']));
    fprintf('%s: validation %.2f%%, external test %.2f%% (%d/%d)\n',bundle.spec.id,100*v.accuracy,100*t.accuracy,t.correct,t.total);
end
leaderboard=sortrows(struct2table(records),{'validationAccuracy','validationMacroF1','model'},{'descend','descend','ascend'});
leaderboard.validationRank=(1:height(leaderboard))';
[~,order]=sort(leaderboard.testAccuracy,'descend'); ranks=zeros(height(leaderboard),1); ranks(order)=1:height(leaderboard);
leaderboard.testRank=ranks;
writetable(leaderboard,fullfile(out,'leaderboard.csv'));
report=struct('schemaVersion',1,'protocolVersion',protocol.version,'fingerprint',data.fingerprint, ...
    'testWindows',numel(data.testLabels),'validationWindows',numel(protocol.validationRows), ...
    'selectionMetric','validationAccuracy','selectedModel',leaderboard.model(1), ...
    'bestObservedTestModel',leaderboard.model(order(1)), ...
    'limitations',{{protocol.limitation,sprintf('External test has %d windows only. Comparison is descriptive, not an unbiased estimate after model selection.',numel(data.testLabels)), ...
    'Algorithms, feature sets, and training fractions differ; results do not isolate the causal effect of algorithm choice.', ...
    'Prototypes are not official submission bundles. Organizer verifier not supplied.'}}, ...
    'leaderboard',table2struct(leaderboard),'details',details);
har.writeJson(fullfile(out,'results.json'),report);
har.renderReport(root,out,report);
end
