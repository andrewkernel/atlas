param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$atlasPython = Join-Path $PSScriptRoot '.tools/attention-env/Scripts/python.exe'
@'
from pathlib import Path
import sys,psutil
root=Path(sys.argv[1]).resolve()
port=int(sys.argv[2])
pids={c.pid for c in psutil.net_connections(kind='tcp') if c.status=='LISTEN' and c.laddr.port==port and c.laddr.ip=='127.0.0.1'}
if not pids:
    print('Atlas is already stopped.');sys.exit(0)
if len(pids)!=1:raise RuntimeError('Ambiguous listener; no process stopped')
process=psutil.Process(pids.pop())
args=process.cmdline()
scripts=[Path(a) if Path(a).is_absolute() else Path(process.cwd())/a for a in args if a.endswith('lab.py')]
if Path(process.cwd()).resolve()!=root or root/'lab.py' not in [p.resolve() for p in scripts] or 'serve' not in args:
    raise RuntimeError('Port belongs to another application; no process stopped')
process.terminate()
process.wait(timeout=10)
print('Atlas local service stopped. Saved experiments are retained.')
'@ | & $atlasPython - $PSScriptRoot $Port
if ($LASTEXITCODE -ne 0) { throw 'Atlas was not stopped; inspect the diagnostic above.' }
