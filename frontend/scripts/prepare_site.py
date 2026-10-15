"""Copy only dashboard assets to the registered private Site's ignored checkout."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT/'frontend'
DESTINATION = ROOT/'.tools/dashboard-site'
FILES = ['index.html','app.css','app.js','data/snapshot.json','assets/atlas.svg',
         'assets/geist.woff2','assets/OFL.txt','assets/font-source.json']


if __name__ == '__main__':
    manifest = json.loads((FRONTEND/'.openai/hosting.json').read_text(encoding='utf-8'))
    prior = json.loads((DESTINATION/'.openai/hosting.json').read_text(encoding='utf-8'))
    assert manifest['project_id'] == prior['project_id']
    assert DESTINATION.resolve().is_relative_to((ROOT/'.tools').resolve())
    (DESTINATION/'.gitattributes').write_bytes(b'* -text\n')
    identities = {}
    for name in FILES:
        src, dst = FRONTEND/name, DESTINATION/'dist'/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
        identities[name] = hashlib.sha256(src.read_bytes()).hexdigest()
    target = DESTINATION/'.openai/hosting.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_bytes((json.dumps(manifest,indent=2)+'\n').encode('utf-8'))
    temporary.replace(target)
    (DESTINATION/'source-identity.json').write_bytes((json.dumps({'source':'andrewkernel/atlas/frontend', 'files':identities},indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'project_id':manifest['project_id'],'assets_copied':len(FILES),'checkout':str(DESTINATION)}))
