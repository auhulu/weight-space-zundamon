"""Freeze and verify inputs before any candidate evaluation."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
paths=[ROOT/'config.json', ROOT/'PROTOCOL.md', *sorted((ROOT/'data').glob('*.json'))]
assert len(list((ROOT/'data').glob('*.json'))) >= 4, 'data incomplete'
# Canonical LF text keeps the frozen content portable across Git autocrlf setups.
hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_text(encoding='utf-8').encode('utf-8')).hexdigest() for p in paths}
path=ROOT/'input_manifest.json'
if path.exists():
    assert json.loads(path.read_text())==hashes, 'frozen experiment inputs changed'
else:
    path.write_text(json.dumps(hashes,indent=2),encoding='utf-8')
print('Verified frozen experiment inputs:',len(hashes))
