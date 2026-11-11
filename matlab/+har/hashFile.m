function digest = hashFile(file)
% ファイルの SHA-256 を、小文字の16進数の文字列で返す。
% データやモデルが差し替わっていないことを確かめるために使う。

chunkBytes = 1024 * 1024;

fileId = fopen(file, 'rb');
assert(fileId ~= -1, 'Cannot read %s', file);
closeFile = onCleanup(@() fclose(fileId));

engine = java.security.MessageDigest.getInstance('SHA-256');
while ~feof(fileId)
    engine.update(fread(fileId, chunkBytes, '*uint8'));
end

hashBytes = typecast(engine.digest(), 'uint8');
digest = lower(reshape(dec2hex(hashBytes, 2)', 1, []));
end
