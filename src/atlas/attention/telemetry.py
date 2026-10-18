"""Passive process/device observations; not a hard residency measurement."""
import subprocess
import threading
import time
import psutil
from .evidence import atomic_json


class Sampler:
    def __init__(self,path=None):
        self.rows=[];self.stop=threading.Event();self.started=time.perf_counter();self.phase='setup'
        self.path=path
        self.process=psutil.Process();self.thread=threading.Thread(target=self.sample,daemon=True)
    def __enter__(self):self.thread.start();return self
    def sample(self):
        while not self.stop.is_set():
            began=time.perf_counter()
            row={'elapsed_s':began-self.started,'phase':self.phase,'rss_mib':self.process.memory_info().rss/2**20,'available_ram_mib':psutil.virtual_memory().available/2**20}
            try:
                values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=3,creationflags=0x08000000).strip().split(',')
                row.update(board_used_mib=float(values[0]),utilization_percent=float(values[1]),temperature_c=float(values[2]))
            except (OSError,ValueError,subprocess.SubprocessError):row['device_observation']='unavailable'
            row['observer_wall_ms']=(time.perf_counter()-began)*1000
            self.rows.append(row);self.stop.wait(.5)
            if self.path:atomic_json(self.path,{'samples':self.rows})
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=4)
