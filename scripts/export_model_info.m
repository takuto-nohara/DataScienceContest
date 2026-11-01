% バイナリモデルを変更せず、UTF-8のJSONと説明文を各モデルに出力する。
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'matlab'));
folders=dir(fullfile(root,'model','proto_*'));folders=folders([folders.isdir]);
for k=1:numel(folders)
    folder=fullfile(folders(k).folder,folders(k).name);
    [bundle,info]=har.inspectModel(folders(k).name,false);
    har.writeJson(fullfile(folder,'model-info.json'),info);
    text=sprintf('# %s：モデルの開き方\n\n`trained.mat` はMATLABのバイナリ形式です。テキストエディターで表示される不規則な文字は、文字コードの破損を示すものではありません。\n\n',bundle.spec.id);
    text=[text sprintf('方式：`%s` ／ 特徴量セット：`%s` ／ 特徴量数：%d個\n\n',bundle.spec.algorithm,bundle.spec.featureSet,numel(bundle.featureNames))];
    text=[text sprintf('MATLABでは、リポジトリのフォルダで次を実行してください。`bundle` が学習済みモデル、`info` が概要です。\n\n```matlab\naddpath(''matlab'');\n[bundle, info] = har.inspectModel(''%s'');\n```\n\n',bundle.spec.id)];
    text=[text 'VSCodeでは [model-info.json](model-info.json) をUTF-8で開くと、方式・設定・特徴量名を確認できます。詳しい学習記録は [manifest.json](manifest.json)、分類の仕組みは [日本語レポート](../../reports/classification-criteria-ja.md) を参照してください。' newline];
    file=fullfile(folder,'README.md');fid=fopen(file,'w','n','UTF-8');assert(fid~=-1);fwrite(fid,unicode2native(text,'UTF-8'),'uint8');fclose(fid);
end
fprintf('MODEL_INFO_EXPORTED: %d models; MAT binaries unchanged\n',numel(folders));
