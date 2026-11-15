function F = productionFeatures(raw)
% 本番の32列（標準化前）を、実験用の関数から作る。前処理関数の検査と標準化の定数に使う。
%   列 1-30  : ergonomicFeatures（回転の揺れ3列は常用対数）
%   列 31-32 : 鉛直軸まわりの角速度（ジャイロx）の1階差分と2階差分の標準偏差
%              （セミナー資料の差分の列のうち、本番の30列に足して効いた2列）
A = araiDiff24(raw);
F = [ergonomicFeatures(raw, true), A(:, [10, 22])];
end
