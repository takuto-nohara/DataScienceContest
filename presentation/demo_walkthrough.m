%% 1. データと学習済みモデルを読み込む
% このファイルをMATLABで開き、セクションごとに実行できます。
% trainは呼びません。既存の学習済みモデルで処理を再現します。
demoRoot=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(demoRoot,'matlab'));
demoData=har.loadData(demoRoot);
demoBundle=har.inspectModel('proto_v10',false);
% 学習時の記録: 元の学習窓2206件、振幅違いを含めた観測6618件。
disp(demoBundle.parameters);
disp(table(numel(demoBundle.fitRows),demoBundle.augmentedRows, ...
    numel(demoBundle.protocol.validationRows),'VariableNames',{'originalFitWindows','augmentedObservations','validationWindows'}));

%% 2. 1件の波形を32特徴量へ変換する
% 155件目は多数決によって主SVMの予測が変わる例です。
demoRow=155;
demoRaw=demoData.testRaw(demoRow,:,:); % 1×128×6
[demoX,demoNames]=har.features(demoRaw,demoBundle.spec.featureSet);
demoZ=(demoX-demoBundle.center)./demoBundle.scale;
disp(table(demoNames(:),demoX(:),demoZ(:),'VariableNames',{'feature','rawValue','z'}));

%% 3. 主SVMと最終判定を比較する
[demoPrimary,demoNegLoss]=predict(demoBundle.estimator.primary,demoZ);
demoFinal=har.predict(demoBundle,demoRaw);
demoExplanation=har.explainWindow(demoBundle,demoRaw);
disp(demoExplanation.treeTable);
disp(demoExplanation.pairTable);
% 正解ラベルは表示・評価にだけ使用し、har.predictへは渡しません。
disp(table(demoData.testLabels(demoRow),string(demoPrimary),demoFinal, ...
    'VariableNames',{'truth','primarySVM','final'}));

%% 4. 図と表で説明する画面を開く
demoApp=har.demo(demoRoot,demoRow);
% 画面上部の番号で1〜201件を切り替えられます。
% 6つのタブを順に開き、必要な箇所で実コードを表示してください。

%% 5. 発表に使う画面をPNGへ保存する（必要な場合に実行）
% demoApp.Export(fullfile(demoRoot,'artifacts','demo'));
