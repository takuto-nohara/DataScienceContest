function F = productionFeatures(raw)
% 本番の32列（標準化前）を、実験用の関数から作る。前処理関数の検査と標準化の定数に使う。
%   列 1-28  : ergonomicFeatures の30列から acc_mag_std と vertical_std を除いた28列
%              （x軸の標準偏差と重なる。第10段で外しても下がらないことを確かめた。
%               回転の揺れ3列は常用対数）
%   列 29-30 : 鉛直軸まわりの角速度（ジャイロx）の1階差分と2階差分の標準偏差
%   列 31-32 : bioPool の pivot_height_mean（骨盤の回転の支点までの距離）と
%              sway_fmean_ap（前後の揺れの平均周波数）
A = araiDiff24(raw);
[B, names] = bioPool(raw);
E = ergonomicFeatures(raw, true);
F = [E(:, [1:7, 10:30]), A(:, [10, 22]), ...
    B(:, names == "pivot_height_mean"), B(:, names == "sway_fmean_ap")];
end
