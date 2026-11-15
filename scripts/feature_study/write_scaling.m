function [featureMean, featureStd] = write_scaling(root)
% 本番の前処理関数に、32列の標準化の定数（平均と標準偏差）を書き込む。
%   定数は学習用2941窓＋拡張8通り（26469行、本番の学習データと同じ）から求める。
%   標準化しない値は productionFeatures（featurePool と差分の2列）で計算する。
%   試行ごとの計算は「その試行の値 − 定数」÷ 定数なので、ほかの試行の値は使わない。
% 分類学習器では「データを標準化」を切って学習する（2クラスの組ごとの標準化をしない）。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'matlab'), here);
data = har.loadData(root);
[featureMean, featureStd] = scalingConstants(data.trainRaw);

file = fullfile(root, 'submission', 'ergonomic32_svm', 'Pretreatment_MATLABcontest_Advanced.m');
text = fileread(file, 'Encoding', 'UTF-8');
fmt = @(v) strjoin(compose('%.17g', v), ', ');
text = regexprep(text, 'featureMean = \[[^\]]*\];', ...
    sprintf('featureMean = [%s];', fmt(featureMean)), 'once');
text = regexprep(text, 'featureStd = \[[^\]]*\];', ...
    sprintf('featureStd = [%s];', fmt(featureStd)), 'once');
fid = fopen(file, 'w', 'n', 'UTF-8');
fwrite(fid, text, 'char');
fclose(fid);
fprintf('SCALING_WRITTEN\n');
end
