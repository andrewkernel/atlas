"""Atlas trusted local runtime tuning CLI and disposable worker entry point."""
from pathlib import Path
import argparse
import json
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))


def main():
    p = argparse.ArgumentParser(description='Measure, verify and reversibly tune Atlas model execution')
    sub = p.add_subparsers(dest='command', required=True)
    serve = sub.add_parser('serve'); serve.add_argument('--port', type=int, default=8765)
    start = sub.add_parser('start'); start.add_argument('--context', type=int, default=2048)
    start.add_argument('--dtype', default='float32'); start.add_argument('--generation-tokens', type=int, default=16)
    start.add_argument('--budget-mib', type=float); start.add_argument('--no-apply', action='store_true')
    start.add_argument('--max-wall-s', type=int, default=14400)
    for name in ('controller', 'status', 'verify', 'cancel', 'inference-worker', 'export'):
        child = sub.add_parser(name); child.add_argument('id')
    worker = sub.add_parser('worker'); worker.add_argument('id'); worker.add_argument('case')
    apply = sub.add_parser('apply'); apply.add_argument('id')
    revert = sub.add_parser('revert'); revert.add_argument('domain')
    gen = sub.add_parser('generate'); gen.add_argument('campaign'); gen.add_argument('--prompt')
    gen.add_argument('--check-native', action='store_true')
    sub.add_parser('profiles')
    args = p.parse_args()
    from atlas.tuning.common import STORE, campaign_dir, domain, identifier, read_json, write_json
    from atlas.tuning import profiles
    if args.command == 'serve':
        import uvicorn
        from atlas.tuning.service import app
        uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')
        return
    if args.command == 'start':
        from atlas.tuning.registry import settings
        config = settings({'context': args.context, 'dtype': args.dtype, 'generation_tokens': args.generation_tokens,
                          'goal': 'budget' if args.budget_mib is not None else 'fastest', 'budget_mib': args.budget_mib,
                          'auto_apply': not args.no_apply, 'max_wall_s': args.max_wall_s})
        ident = 'tune-' + uuid.uuid4().hex[:12]
        folder = campaign_dir(ident); folder.mkdir(parents=True)
        write_json(folder / 'request.json', config)
        print(json.dumps({'id': ident}), flush=True)
        from atlas.tuning.controller import run_campaign
        result = run_campaign(ident)
    elif args.command == 'controller':
        from atlas.tuning.controller import run_campaign
        result = run_campaign(identifier(args.id))
    elif args.command == 'worker':
        from atlas.tuning.executor import run_worker
        result = run_worker(campaign_dir(args.id), identifier(args.case))
    elif args.command == 'inference-worker':
        from atlas.tuning.inference import run
        result = run(identifier(args.id))
    elif args.command == 'status':
        result = read_json(campaign_dir(args.id) / 'status.json')
    elif args.command == 'verify':
        from atlas.tuning.verify import build
        result = build(campaign_dir(args.id))
    elif args.command == 'cancel':
        result = {'status': profiles.cancel(identifier(args.id))}
        (campaign_dir(args.id) / 'cancel').touch()
    elif args.command == 'profiles':
        result = {'profiles': [p.stem for p in (STORE / 'profiles').glob('*.json')], 'events': profiles.history()}
    elif args.command == 'apply':
        value = profiles.read(args.id)
        result = profiles.activate(args.id, profiles.state(value['domain']))
    elif args.command == 'revert':
        result = profiles.revert(args.domain, profiles.state(args.domain))
    elif args.command == 'generate':
        protocol = read_json(campaign_dir(args.campaign) / 'protocol.json')
        ident = 'generate-' + uuid.uuid4().hex[:12]
        folder = STORE / 'private-inference' / ident
        write_json(folder / 'request.json', {'settings': protocol['settings'], 'example_campaign': args.campaign,
                                          'prompt': args.prompt, 'check_native': args.check_native})
        from atlas.tuning.inference import run
        result = run(ident)
    elif args.command == 'export':
        from atlas.tuning.bundle import export
        result = {'path': str(export(identifier(args.id)))}
    else:
        raise ValueError('Unregistered command')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
