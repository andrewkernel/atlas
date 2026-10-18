"""Identify the exact checkout's listener before reuse, upgrade or stop."""
from pathlib import Path
import argparse
import json
import urllib.request
import psutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path); p.add_argument('port', type=int)
    p.add_argument('--stop', action='store_true'); p.add_argument('--require-idle', action='store_true')
    args = p.parse_args(); root = args.root.resolve()
    pids = {c.pid for c in psutil.net_connections(kind='tcp') if c.status == 'LISTEN'
            and c.laddr.port == args.port and c.laddr.ip == '127.0.0.1'}
    if not pids:
        print(json.dumps({'found': False})); return
    if len(pids) != 1:
        raise RuntimeError('Ambiguous listener; no process changed')
    process = psutil.Process(pids.pop()); created = process.create_time(); command = process.cmdline()
    scripts = [Path(a) if Path(a).is_absolute() else Path(process.cwd()) / a for a in command
               if a.endswith(('lab.py', 'tune.py'))]
    if (Path(process.cwd()).resolve() != root or 'serve' not in command or len(scripts) != 1
            or scripts[0].resolve() not in (root / 'lab.py', root / 'tune.py')):
        raise RuntimeError('Port belongs to another application or checkout; no process changed')
    result = {'found': True, 'owned': True, 'entrypoint': scripts[0].name, 'pid': process.pid}
    if args.stop:
        if args.require_idle:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/session', timeout=3) as response:
                session = json.load(response)
            if session.get('active_run') or session.get('tuning_active'):
                raise RuntimeError('Atlas has an active job; finish or cancel it before upgrading the service')
        if process.create_time() != created:
            raise RuntimeError('Listener identity changed; no process stopped')
        process.terminate(); process.wait(timeout=10)
        result['stopped'] = True
    print(json.dumps(result))


if __name__ == '__main__':
    main()
