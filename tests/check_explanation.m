root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'matlab'));
d=har.loadData(root);b=har.inspectModel('proto_v10',false);
expected=har.predict(b,d.testRaw);changed=[];
for i=1:numel(expected)
    e=har.explainWindow(b,d.testRaw(i,:,:));
    assert(e.final==expected(i) && height(e.featureTable)==32 && height(e.pairTable)==15);
    [~,winner]=max(e.negativeLoss);assert(e.classes(winner)==e.primary);
    assert(all(isfinite(e.standardized)) && numel(e.votes)==3);
    assert(max(abs(e.standardized-(e.features-e.center)./e.scale))<1e-12);
    assert(e.treeLabel==e.votes(2));
    if e.primary~=e.final,changed(end+1)=i;end %#ok<SAGROW>
end
assert(isequal(changed,155));
assert(e.final==expected(end));
app=har.demo(root,155);cleanup=onCleanup(@()delete(app.Figure));
assert(app.GetExplanation().final=="Sitting");
app.SelectRow(1);assert(app.RowPicker.Value==1 && app.GetExplanation().final==expected(1));
app.SelectRow(155);app.Export(fullfile(root,'artifacts','demo'));
assert(numel(dir(fullfile(root,'artifacts','demo','row155_step*.png')))==6);
fprintf('EXPLANATION_CHECK_PASSED: 201 exact traces, 15 SVM pairs, saved-tree paths, row switching, six PNGs\n');
