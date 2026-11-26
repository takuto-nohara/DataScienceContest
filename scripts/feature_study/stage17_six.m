function stage17_six(root, repeats)
% 第17段：後ろ向き選択で最後に残った6列（2 3 10 18 25 32）で、各列の役割を確かめる。
%   ・6列から1列ずつ外した5列の組
%   ・躍度z（列18）を、近い情報を持つ列に入れ替えた組
%     加速度zの標準偏差（7）、躍度x（16）、躍度y（17）、上下方向の躍度の最大（19）・最小（20）
%   測り方は stage16_industrial と同じ。
if nargin < 1 || isempty(root)
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2, repeats = 3; end
base = [2 3 10 18 25 32];
label = containers.Map({2, 3, 10, 18, 25, 32}, ...
    {'重力の向きy', '重力の向きz', 'xy方向の揺れ', '躍度z', '上下と水平の相関', 'y方向の揺れの周波数'});
setNames = "6列（基準）";
sets = {base};
for c = base
    setNames(end + 1) = "外す：" + label(c); %#ok<AGROW>
    sets{end + 1} = setdiff(base, c, 'stable'); %#ok<AGROW>
end
swaps = [7 16 17 19 20];
swapNames = ["加速度zの標準偏差", "躍度x", "躍度y", "上下方向の躍度の最大", "上下方向の躍度の最小"];
for k = 1:numel(swaps)
    setNames(end + 1) = "躍度z→" + swapNames(k); %#ok<AGROW>
    s = base; s(s == 18) = swaps(k);
    sets{end + 1} = s; %#ok<AGROW>
end
stage16_industrial(root, repeats, setNames, sets, 'stage17_six.csv');
end
