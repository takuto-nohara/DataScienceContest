% このファイル自体をUTF-8で保存し、MATLABが日本語を正しく読めることを確認する。
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'matlab'));
expected='日本語の表示確認';
fprintf('%s\n',expected);
bytes=unicode2native(expected,'UTF-8');assert(isequal(native2unicode(bytes,'UTF-8'),expected));
report=fileread(fullfile(root,'reports','classification-criteria-ja.md'));
assert(contains(report,'行動認識モデルは何を基準に分類しているか'));
output=fullfile(root,'artifacts','encoding-japanese.csv');
fixture=table(["歩行";"立位"],["正解";"正解"],'VariableNames',{'activity','result'});
writetable(fixture,output,'Encoding','UTF-8');assert(contains(fileread(output),'歩行'));
output=fullfile(root,'artifacts','encoding-japanese.json');
har.writeJson(output,struct('message',expected));decoded=jsondecode(fileread(output));assert(strcmp(decoded.message,expected));
folders=dir(fullfile(root,'model','proto_*','trained.mat'));
for k=1:numel(folders)
    file=fullfile(folders(k).folder,folders(k).name);loaded=load(file,'bundle');
    assert(isstruct(loaded.bundle) && ~isempty(loaded.bundle.featureNames));
    manifest=jsondecode(fileread(fullfile(folders(k).folder,'manifest.json')));
    assert(strcmp(manifest.artifactSHA256,har.hashFile(file)),'Model binary was modified.');
    assert(contains(fileread(fullfile(folders(k).folder,'README.md')),'モデルの開き方'));
end
fprintf('UTF8_AND_MAT_CHECK_PASSED: Japanese source/report, %d model binaries\n',numel(folders));
