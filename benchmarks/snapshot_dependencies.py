from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
from urllib.request import urlopen
from packaging.specifiers import SpecifierSet

names=['torch','numpy','scipy','pytest','hypothesis','psutil','uv','fastapi','pydantic','mlflow','nvidia-ml-py']
def inspect(name):
    try:
        installed=version(name)
    except PackageNotFoundError:
        installed=None
    endpoint=f'https://pypi.org/pypi/{name}/'+(installed.split('+')[0]+'/' if installed else '')+'json'
    try:
        with urlopen(endpoint,timeout=30) as response:
            info=json.load(response)['info']
        requires=info.get('requires_python') or ''
        return {'name':name,'installed_version':installed,'metadata_version':info['version'],'requires_python':requires,'python_3_12_10_admitted_by_metadata':bool('3.12.10' in SpecifierSet(requires)) if requires else None,'license_expression':info.get('license_expression'),'license_classifiers':[c for c in info.get('classifiers',[]) if c.startswith('License ::')],'license_metadata_excerpt':(info.get('license') or '')[:180],'source_url':endpoint,'status':'installed in CPU environment; package-specific execution is recorded in the research log' if installed else 'metadata only; not installed or integration-tested'}
    except Exception as exc:
        return {'name':name,'installed_version':installed,'source_url':endpoint,'error':str(exc)}
with ThreadPoolExecutor(max_workers=4) as pool:
    records=list(pool.map(inspect,names))
output={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'caveat':'Package metadata is not a full license/dependency compatibility audit. CPU wheel source is separately pinned in uv.lock. Optional stack components are not validated by metadata.','packages':records}
Path('docs/research/dependency-snapshot.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
for item in records:
    print(item['name'],item.get('installed_version'),item.get('metadata_version'),item.get('requires_python'),item.get('license_expression'),item.get('license_classifiers'),item.get('error',''))
