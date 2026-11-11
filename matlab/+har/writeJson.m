function writeJson(file, value)
% 値を、読みやすく整形した UTF-8 の JSON ファイルとして書き出す。
% 書き出し先のフォルダがなければ作る。

folder = fileparts(file);
if ~isfolder(folder)
    mkdir(folder);
end

fileId = fopen(file, 'w', 'n', 'UTF-8');
assert(fileId ~= -1, 'Cannot write %s', file);
closeFile = onCleanup(@() fclose(fileId));

fprintf(fileId, '%s\n', jsonencode(value, PrettyPrint=true));
end
