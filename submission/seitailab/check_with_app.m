function check_with_app()
% 提出前チェックアプリ（DScontest_submitModelVerifier）本体で、提出ファイルを確かめる
%
% 配布された .mltbx を一時フォルダに展開し、アプリをインストールせずに起動する。
% Step1〜3 のボタンを押したときと同じコールバックを順に呼ぶ。
% ファイル選択の画面（uigetfile）だけは、提出ファイルを返す関数に差し替える。
% 結果の画面は check_app_result.png に保存する。

here = fileparts(mfilename('fullpath'));
mltbx = fullfile(here, '..', '..', 'data', 'production', 'DScontest_submitModelVerifier.mltbx');
work = tempname;
unzip(mltbx, work);                                  % .mltbx は zip 形式
stubDir = fullfile(work, 'stub');
mkdir(stubDir);
fid = fopen(fullfile(stubDir, 'uigetfile.m'), 'w');
fprintf(fid, '%s\n', ...
    'function [file, folder] = uigetfile(varargin)', ...
    '[folder, name, ext] = fileparts(getappdata(groot, ''stubPick''));', ...
    'file = [name ext]; folder = [folder filesep];', ...
    'end');
fclose(fid);
addpath(fullfile(work, 'fsroot'));
addpath(stubDir, '-begin');
cleanup = onCleanup(@() rmpath(stubDir, fullfile(work, 'fsroot')));

app = DScontest_submitModelVerifier;
pick = @(f) setappdata(groot, 'stubPick', fullfile(here, f));
push = @(b) b.ButtonPushedFcn(b, []);

pick('Pretreatment_MATLABcontest_Advanced_seitailab.m');
push(app.FileSelectButton_step1);
report('Step1', app.FilenameReadStatus_step1, app.FilenameReadStatusLamp_step1);

pick('trainedModel_seitailab.mat');
push(app.FileSelectButton_step2);
report('Step2', app.FilenameReadStatus_step2, app.FilenameReadStatusLamp_step2);

push(app.isCanVerifyButton_step3);
report('Step3', app.ShowResultText_step3, app.CanVerifyStatusLamp_step3);

drawnow;
exportapp(app.UIFigure, fullfile(here, 'check_app_result.png'));
delete(app);
end

function report(step, text, lamp)
names = containers.Map({'0 1 0', '1 1 0', '1 0 0'}, {'緑', '黄', '赤'});
key = num2str(lamp.Color);
key = regexprep(strtrim(key), '\s+', ' ');
if isKey(names, key), c = names(key); else, c = key; end
fprintf('%s [%s] %s\n', step, c, strjoin(string(text.Value), ' '));
end
