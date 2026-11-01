function labels = staticVote(primary, votes)
primary=string(primary(:));votes=string(votes);assert(size(votes,1)==numel(primary));labels=primary;
for row=1:numel(primary)
    if ~ismember(primary(row),["Sitting","Standing","Laying"]),continue;end
    for column=1:size(votes,2)
        if sum(votes(row,:)==votes(row,column))>size(votes,2)/2
            labels(row)=votes(row,column);break;
        end
    end
end
end
