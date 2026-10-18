"""Immutable evidence-bound profiles and serialized activation/cancellation."""
from __future__ import annotations

import sqlite3

from .common import (ROOT, STORE, campaign_dir, check_sources, digest, domain, frozen_json,
                     identifier, read_json, runtime_identity, utc)


class Conflict(RuntimeError):
    pass


def connect(home=STORE):
    home.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(home / 'state.sqlite3', timeout=15)
    db.row_factory = sqlite3.Row
    db.executescript('''
        CREATE TABLE IF NOT EXISTS active(domain TEXT PRIMARY KEY, profile TEXT, revision INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, domain TEXT, before_profile TEXT,
          after_profile TEXT, revision INTEGER, action TEXT, campaign TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS decisions(campaign TEXT PRIMARY KEY, state TEXT NOT NULL, profile TEXT);
    ''')
    return db


def state(key, home=STORE):
    with connect(home) as db:
        row = db.execute('SELECT profile, revision FROM active WHERE domain=?', (key,)).fetchone()
    return dict(row) if row else {'profile': None, 'revision': 0}


def history(home=STORE):
    with connect(home) as db:
        return [dict(r) for r in db.execute('SELECT * FROM events ORDER BY id DESC LIMIT 100')]


def cancel(campaign, home=STORE):
    identifier(campaign)
    with connect(home) as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT state FROM decisions WHERE campaign=?', (campaign,)).fetchone()
        if row and row['state'] == 'committed':
            raise Conflict('Decision already committed; use Revert to change applied settings')
        db.execute('INSERT OR IGNORE INTO decisions VALUES (?, ?, NULL)', (campaign, 'cancelled'))
    return 'cancelled'


def cancelled(campaign, home=STORE):
    with connect(home) as db:
        row = db.execute('SELECT state FROM decisions WHERE campaign=?', (campaign,)).fetchone()
    return bool(row and row['state'] == 'cancelled')


def save(folder, verified, home=STORE):
    protocol = read_json(folder / 'protocol.json')
    chosen = verified['decision']['profile']
    if chosen is None or verified['decision']['outcome'] not in (
            'confirmed_improvement', 'confirmed_memory_fit', 'keep_current'):
        return None
    profile = {'schema_version': 1, 'campaign': folder.name, 'domain': domain(protocol['settings']),
               'settings': protocol['settings'], 'identity': protocol['identity'],
               'sources': protocol['sources'], 'protocol_sha256': verified['protocol_sha256'],
               'evidence_sha256': verified['evidence_sha256'], 'decision': verified['decision'],
               'execution': next(p for p in protocol['profiles'] if p['id'] == chosen)}
    ident = 'profile-' + digest(profile)[:24]
    frozen_json(home / 'profiles' / (ident + '.json'), profile)
    return ident


def read(ident, home=STORE):
    identifier(ident)
    value = read_json(home / 'profiles' / (ident + '.json'))
    if ident != 'profile-' + digest(value)[:24]:
        raise ValueError('Immutable profile identity differs')
    return value


def validate(ident, home=STORE, root=ROOT, current_identity=True):
    from .verify import build
    profile = read(ident, home)
    check_sources(profile['sources'], root)
    if current_identity and profile['identity'] != runtime_identity():
        raise ValueError('Profile is incompatible with the current hardware, driver or packages')
    folder = campaign_dir(profile['campaign'], home)
    verified = build(folder, root)
    protocol = read_json(folder / 'protocol.json')
    if (profile['protocol_sha256'] != verified['protocol_sha256'] or
            profile['evidence_sha256'] != verified['evidence_sha256'] or
            profile['decision'] != verified['decision'] or
            profile['settings'] != protocol['settings'] or profile['identity'] != protocol['identity'] or
            profile['domain'] != domain(protocol['settings']) or
            profile['execution'] != next(p for p in protocol['profiles'] if p['id'] == verified['decision']['profile'])):
        raise ValueError('Profile does not match independently reconstructed campaign evidence')
    return profile


def transition(key, target, expected, action, campaign=None, home=STORE):
    """Only called after verification. No long model/hash operation inside this transaction."""
    with connect(home) as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT profile, revision FROM active WHERE domain=?', (key,)).fetchone()
        current = dict(row) if row else {'profile': None, 'revision': 0}
        if current != expected:
            raise Conflict('Active profile changed during this operation; review current state')
        if campaign:
            row = db.execute('SELECT state FROM decisions WHERE campaign=?', (campaign,)).fetchone()
            if row:
                raise Conflict('Campaign decision is already ' + row['state'])
        revision = current['revision'] + 1
        db.execute('INSERT OR REPLACE INTO active VALUES (?, ?, ?)', (key, target, revision))
        db.execute('INSERT INTO events(domain,before_profile,after_profile,revision,action,campaign,created_at) '
                   'VALUES (?,?,?,?,?,?,?)', (key, current['profile'], target, revision, action, campaign, utc()))
        if campaign:
            db.execute('INSERT INTO decisions VALUES (?, ?, ?)', (campaign, 'committed', target))
    return {'profile': target, 'revision': revision}


def activate(ident, expected, home=STORE, campaign=None, root=ROOT, before_commit=None):
    profile = validate(ident, home, root)
    if before_commit:
        before_commit()
    return transition(profile['domain'], ident, expected, 'automatic' if campaign else 'apply', campaign, home)


def revert(key, expected, home=STORE, root=ROOT):
    with connect(home) as db:
        row = db.execute('SELECT before_profile FROM events WHERE domain=? AND revision=?',
                         (key, expected['revision'])).fetchone()
    if not row:
        raise Conflict('There is no activation to revert')
    target = row['before_profile']
    if target:
        validate(target, home, root)
    return transition(key, target, expected, 'revert', home=home)
