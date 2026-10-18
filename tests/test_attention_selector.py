import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atlas.attention.selector import ExecutionKVPolicy,execution_kv_mib


def test_copy_to_gpu_is_not_a_zero_gpu_memory_plan():
    identity={'dtype':'float32','batch':1,'kv_heads':8,'dim':128,'scope':'layer-attend-only'}
    assert execution_kv_mib('copy',20480,identity)==160
    assert execution_kv_mib('hybrid25',20480,identity)==40
    low={**identity,'dtype':'float16'}
    assert execution_kv_mib('gpu_dense',20480,low)==240
    rows=[{'phase':'calibration','context':n,'plan':p,'median_ms':ms,'identity':identity,'gpu_kv_mib':0,'correctness':{'passed':True}} for n in (768,32768) for p,ms in [('copy',1),('cpu',3)]]
    policy=ExecutionKVPolicy(rows,identity)
    assert policy.choose(20480,identity,48)['plan']=='cpu'
    assert policy.choose(20480,identity,2048)['plan']=='copy'
    assert policy.choose(20480,{**identity,'scope':'full-model'},2048)['abstained']


def test_one_failed_calibration_blocks_interpolation_across_it():
    identity={'dtype':'float32','batch':1,'kv_heads':8,'dim':128,'scope':'layer-attend-only'}
    rows=[{'phase':'calibration','context':n,'plan':p,'median_ms':ms,'identity':identity,'gpu_kv_mib':0,'correctness':{'passed':not (p=='resident' and n==500)}} for n in (100,500,1000) for p,ms in [('resident',1),('cpu',3)]]
    assert ExecutionKVPolicy(rows,identity).choose(800,identity,2048)['plan']=='cpu'
