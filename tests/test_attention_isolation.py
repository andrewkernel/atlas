"""Invented numerical fault controls, never model or hardware evidence."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from verify_attention_v3 import compare_arrays
from verify_attention_isolation import check_saved_decisions


def test_isolated_verifier_refuses_a_forged_success_after_array_change():
    reference=np.array([[1.,2.,3.],[3.,2.,1.]],dtype=np.float32)
    contract={'atol':1e-4,'rtol':1e-4}
    row={'logit_checks':compare_arrays(reference,reference,**contract),'correctness':{'passed':True}}
    assert check_saved_decisions(reference,reference,row,contract)==(True,0)
    changed=reference.copy();changed[1,2]+=.1
    with pytest.raises(ValueError,match='disagrees with arrays'):
        check_saved_decisions(changed,reference,row,contract)


def test_isolated_verifier_never_promotes_nonfinite_or_missing_positions():
    reference=np.ones((2,3),dtype=np.float32);actual=reference.copy();actual[0,1]=np.nan
    contract={'atol':1e-4,'rtol':1e-4}
    row={'logit_checks':compare_arrays(actual,reference,**contract),'correctness':{'passed':False}}
    assert check_saved_decisions(actual,reference,row,contract)[0] is False
    row['logit_checks'].pop()
    with pytest.raises(ValueError,match='Incomplete isolated'):
        check_saved_decisions(actual,reference,row,contract)
