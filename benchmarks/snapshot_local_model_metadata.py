"""Read-only first-party model/config/registry metadata; never downloads weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

MODELS = ['Qwen/Qwen3.5-4B', 'Qwen/Qwen3-4B', 'Qwen/Qwen2.5-Coder-7B-Instruct']


def fetch_json(url: str) -> tuple[dict, bytes]:
    with urlopen(Request(url, headers={'Accept': 'application/json'}), timeout=30) as response:
        body = response.read()
    return json.loads(body), body


def run() -> dict:
    models = []
    for name in MODELS:
        api_url = 'https://huggingface.co/api/models/'+name
        metadata, _ = fetch_json(api_url)
        config_url = 'https://huggingface.co/'+name+'/resolve/'+metadata['sha']+'/config.json'
        config, body = fetch_json(config_url)
        text = config.get('text_config', config)
        full_layers = text.get('layer_types', ['full_attention']*text['num_hidden_layers']).count('full_attention')
        head_dim = text.get('head_dim', text['hidden_size']//text['num_attention_heads'])
        kv_per_token = 2*full_layers*text['num_key_value_heads']*head_dim*2
        models.append({'id': name, 'revision': metadata['sha'], 'license_metadata': metadata.get('cardData', {}).get('license'),
                       'safetensors_metadata': metadata.get('safetensors'), 'api_url': api_url,
                       'config_url': config_url, 'config_sha256': hashlib.sha256(body).hexdigest(), 'config': config,
                       'analytical_full_attention_kv_bytes_per_token_at_2_bytes_per_element': kv_per_token,
                       'analytical_full_attention_kv_bytes': {str(tokens): kv_per_token*tokens for tokens in (4096,8192,32768,262144)}})
    registry = 'https://registry.ollama.ai/v2/library/qwen3.5/'
    request = Request(registry+'manifests/4b', headers={'Accept': 'application/vnd.docker.distribution.manifest.v2+json'})
    with urlopen(request, timeout=30) as response:
        body = response.read()
    manifest = json.loads(body)
    config, _ = fetch_json(registry+'blobs/'+manifest['config']['digest'])
    return {'schema_version': '1.0', 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'models': models,
            'ollama_registry': {'manifest_url': registry+'manifests/4b', 'raw_manifest_sha256': hashlib.sha256(body).hexdigest(),
                                'manifest': manifest, 'config': config,
                                'all_layer_payload_bytes': sum(layer['size'] for layer in manifest['layers'])},
            'limitation': 'Metadata only; no model/projector blobs downloaded or executed. KV formula is analytical, batch/concurrency one, two-byte elements; excludes hybrid recurrent state, weights, quantization overhead and workspaces. Context rows do not imply model support or fit.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'models': [model['id'] for model in result['models']],
                      'ollama_manifest_sha256': result['ollama_registry']['raw_manifest_sha256'],
                      'ollama_layer_bytes': result['ollama_registry']['all_layer_payload_bytes']}, indent=2))
