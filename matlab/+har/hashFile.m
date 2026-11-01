function digest = hashFile(file)
fid = fopen(file, 'rb');
assert(fid ~= -1, 'Cannot read %s', file);
cleanup = onCleanup(@() fclose(fid));
engine = java.security.MessageDigest.getInstance('SHA-256');
while ~feof(fid)
    engine.update(fread(fid, 1024*1024, '*uint8'));
end
digest = lower(reshape(dec2hex(typecast(engine.digest(), 'uint8'), 2)', 1, []));
end
