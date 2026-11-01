function writeJson(file, value)
folder = fileparts(file);
if ~isfolder(folder), mkdir(folder); end
fid = fopen(file, 'w', 'n', 'UTF-8');
assert(fid ~= -1, 'Cannot write %s', file);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(value, PrettyPrint=true));
end
