"""Actual cancellation boundary on Windows; skip unsupported OS explicitly."""
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest
from benchmarks.windows_job_scope import WindowsJobScope


@pytest.mark.skipif(os.name != 'nt', reason='Windows job boundary requires Windows')
@pytest.mark.parametrize('action', ['terminate', 'close'])
def test_owned_launcher_and_interpreter_exit_as_a_group(tmp_path, action):
    ready = tmp_path / 'ready.txt'
    # Pass literal data as an argument; no shell and no interpolated Python program.
    command = [sys.executable, '-c',
               'import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text("ready"); time.sleep(5)', str(ready)]
    job = WindowsJobScope()
    launch = subprocess.Popen(command, creationflags=0x00000004, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    root = psutil.Process(launch.pid)
    observed = [root]
    try:
        with pytest.raises(ValueError):
            job.assign_owned(root.pid, root.create_time() + 1)
        with pytest.raises(ValueError):
            job.resume_owned_single_thread(root)
        job.assign_owned(root.pid, root.create_time())
        assert job.resume_owned_single_thread(root) == 1
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline and launch.poll() is None:
            time.sleep(.01)
        assert ready.exists()
        observed = [psutil.Process(pid) for pid in job.members()]
        assert root.pid in {p.pid for p in observed}
        if action == 'terminate':
            job.terminate(46)
        else:
            job.close()
        launch.wait(timeout=3)
        _, alive = psutil.wait_procs(observed, timeout=2)
        assert not alive
        if action == 'terminate':
            assert launch.returncode == 46 and job.members() == []
        job.close()
        with pytest.raises(RuntimeError):
            job.members()
    finally:
        if job.handle is not None:
            job.terminate(47)
            job.close()
        for process in observed:
            if process.is_running():
                process.terminate()
        launch.wait(timeout=3)
