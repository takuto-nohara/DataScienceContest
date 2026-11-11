function labels = staticVote(primary, votes)
% 主モデルが静止姿勢と判定した試行だけ、複数モデルの多数決で判定し直す。
%
%   primary : N×1 の主モデルの判定
%   votes   : N×M の各モデルの判定 (主モデルの判定も1列として含める)
%   labels  : N×1 の最終判定
%
% 過半数を得た行動名があればそれを採用する。票が割れた場合と、主モデルが
% 歩行などの動作と判定した場合は、主モデルの判定をそのまま使う。

staticClasses = ["Sitting", "Standing", "Laying"];

primary = string(primary(:));
votes = string(votes);
assert(size(votes, 1) == numel(primary));

voterCount = size(votes, 2);
labels = primary;

for row = 1:numel(primary)
    if ~ismember(primary(row), staticClasses)
        continue
    end

    for voter = 1:voterCount
        candidate = votes(row, voter);
        hasMajority = sum(votes(row, :) == candidate) > voterCount / 2;
        if hasMajority
            labels(row) = candidate;
            break
        end
    end
end
end
