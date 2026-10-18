"""Allowlisted public assets, pinned revisions, streaming identities; no remote code."""
from pathlib import Path
import hashlib
import json
from .evidence import ROOT,atomic_json

MODELS={
    'qwen3-0.6b':{'repo':'Qwen/Qwen3-0.6B','revision':'c1899de289a04d12100db370d81485cdf75e47ca','license':'Apache-2.0'},
    'qwen3-4b':{'repo':'Qwen/Qwen3-4B','revision':'1cfa9a7208912126459214e8b04321603b3df60c','license':'Apache-2.0'},
}
DATASET={'repo':'Salesforce/wikitext','revision':'b08601e04326c79dfdd32d625aee71d232d685c3','subset':'wikitext-2-raw-v1','license':'CC-BY-SA-3.0 / GFDL'}


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def model_dir(name):
    if name not in MODELS:raise ValueError('Unregistered model')
    return ROOT/'weights'/name


def prepare(name):
    from huggingface_hub import snapshot_download
    info=MODELS[name];folder=model_dir(name)
    snapshot_download(info['repo'],revision=info['revision'],local_dir=folder,max_workers=2,
        allow_patterns=['config.json','generation_config.json','*.safetensors','*.safetensors.index.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','README.md','LICENSE'])
    manifest={**info,'files_sha256':{p.name:file_sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='atlas-manifest.json'},
        'loader':'builtin Transformers, local safetensors only, trust_remote_code=False'}
    atomic_json(folder/'atlas-manifest.json',manifest)
    print(json.dumps({'prepared':name,'revision':info['revision'],'files':len(manifest['files_sha256'])}))


def prepare_corpus():
    from huggingface_hub import hf_hub_download
    folder=ROOT/'data'/'wikitext-attention-v2';folder.mkdir(parents=True,exist_ok=True)
    files={}
    for split in ('train','validation','test'):
        relative=f"{DATASET['subset']}/{split}-00000-of-00001.parquet"
        path=Path(hf_hub_download(DATASET['repo'],relative,repo_type='dataset',revision=DATASET['revision'],local_dir=folder))
        files[relative]=file_sha(path)
    atomic_json(folder/'atlas-manifest.json',{**DATASET,'files_sha256':files,
        'attribution':'WikiText by Stephen Merity et al.; source text from Wikipedia. See https://huggingface.co/datasets/Salesforce/wikitext'})
    print(json.dumps({'prepared':'wikitext','revision':DATASET['revision'],'files':len(files)}))


def corpus_text(split):
    if split not in ('train','validation','test'):raise ValueError('Unknown corpus split')
    import pyarrow.parquet as pq
    folder=ROOT/'data'/'wikitext-attention-v2'
    manifest=json.loads((folder/'atlas-manifest.json').read_text())
    relative=f"{DATASET['subset']}/{split}-00000-of-00001.parquet"
    path=folder/relative
    if file_sha(path)!=manifest['files_sha256'][relative]:raise ValueError('Corpus bytes differ from pinned manifest')
    rows=pq.read_table(path,columns=['text']).column('text').to_pylist()
    return '\n\n'.join(text for text in rows if text and text.strip()),manifest
