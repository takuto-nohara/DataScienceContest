"""Validate repository text as UTF-8 and model binaries against their manifests."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(
    ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT
).decode('utf-8').split('\0')
text_suffixes = {'.m', '.md', '.json', '.csv', '.html', '.py', '.ps1'}
count = 0
for name in sorted(set(filter(None, paths))):
    path = ROOT / name
    if path.suffix.lower() not in text_suffixes and path.name not in {'.editorconfig', '.gitattributes', '.gitignore'}:
        continue
    data = path.read_bytes()
    assert not data.startswith(b'\xef\xbb\xbf'), f'Unexpected UTF-8 BOM: {name}'
    text = data.decode('utf-8', errors='strict')
    assert '\ufffd' not in text, f'Unicode replacement character: {name}'
    count += 1
models = list((ROOT / 'model').glob('proto_*/trained.mat'))
for path in models:
    data = path.read_bytes()
    assert data.startswith(b'MATLAB 5.0 MAT-file'), path
    manifest = json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(data).hexdigest() == manifest['artifactSHA256'], path
    info = json.loads(path.with_name('model-info.json').read_text(encoding='utf-8'))
    assert info['artifactSHA256'] == manifest['artifactSHA256']
settings = json.loads((ROOT / '.vscode/settings.json').read_text(encoding='utf-8'))
assert settings['files.encoding'] == 'utf8' and settings['files.autoGuessEncoding'] is False
print(f'ENCODING_AUDIT_PASSED: {count} UTF-8 text files, {len(models)} unchanged MAT binaries and readable JSON summaries')
