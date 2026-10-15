"""Narrow trusted-worker Windows job boundary; not a hostile-code sandbox."""
import ctypes
from ctypes import wintypes as W
import os


class BasicLimits(ctypes.Structure):
    _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                ('flags', W.DWORD), ('minimum_ws', ctypes.c_size_t), ('maximum_ws', ctypes.c_size_t),
                ('active_process_limit', W.DWORD), ('affinity', ctypes.c_size_t),
                ('priority_class', W.DWORD), ('scheduling_class', W.DWORD)]


class IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic', BasicLimits), ('io', IOCounters), ('process_memory_limit', ctypes.c_size_t),
                ('job_memory_limit', ctypes.c_size_t), ('peak_process_memory', ctypes.c_size_t),
                ('peak_job_memory', ctypes.c_size_t)]


class WindowsJobScope:
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('Windows job profile is unavailable on this OS')
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        bindings = {
            'CreateJobObjectW': ([ctypes.c_void_p, W.LPCWSTR], W.HANDLE),
            'SetInformationJobObject': ([W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD], W.BOOL),
            'QueryInformationJobObject': ([W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD, ctypes.c_void_p], W.BOOL),
            'OpenProcess': ([W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
            'GetProcessTimes': ([W.HANDLE] + [ctypes.POINTER(W.FILETIME)] * 4, W.BOOL),
            'AssignProcessToJobObject': ([W.HANDLE, W.HANDLE], W.BOOL),
            'TerminateJobObject': ([W.HANDLE, W.UINT], W.BOOL),
            'CloseHandle': ([W.HANDLE], W.BOOL),
        }
        for name, (arguments, result) in bindings.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = arguments, result
        self.handle = self.api.CreateJobObjectW(None, None)
        self._check(self.handle)
        try:
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway flags.
            self._check(self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    def _open(self):
        if self.handle is None:
            raise RuntimeError('Job scope is closed')
        return self.handle

    def assign_owned(self, pid, expected_creation_time):
        """Caller must first discover PID in its owned tree; verify the opened handle's identity."""
        job = self._open()
        process = self._check(self.api.OpenProcess(0x0100 | 0x0001 | 0x1000, False, pid))
        try:
            creation, exited, kernel, user = (W.FILETIME() for _ in range(4))
            self._check(self.api.GetProcessTimes(process, ctypes.byref(creation), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)))
            ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
            observed_creation = ticks / 10_000_000 - 11_644_473_600
            if abs(observed_creation - expected_creation_time) > 1e-5:
                raise ValueError('Opened process creation identity differs from retained owned identity')
            self._check(self.api.AssignProcessToJobObject(job, process))
        finally:
            self._check(self.api.CloseHandle(process))

    def members(self):
        job = self._open()
        for capacity in (16, 64, 256):
            class ProcessList(ctypes.Structure):
                _fields_ = [('assigned', W.DWORD), ('listed', W.DWORD), ('pids', ctypes.c_size_t * capacity)]
            values = ProcessList()
            if self.api.QueryInformationJobObject(job, 3, ctypes.byref(values), ctypes.sizeof(values), None):
                if values.listed > capacity:
                    raise RuntimeError('Invalid job process list length')
                return list(values.pids[:values.listed])
            if ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                self._check(False)
        raise RuntimeError('Job process inventory exceeds trusted diagnostic bound')

    def memory_accounting(self):
        values = ExtendedLimits()
        self._check(self.api.QueryInformationJobObject(self._open(), 9, ctypes.byref(values), ctypes.sizeof(values), None))
        return {'peak_job_commit_bytes': values.peak_job_memory,
                'peak_process_commit_bytes': values.peak_process_memory,
                'configured_limit_flags': values.basic.flags,
                'hard_process_memory_limit_bytes': None, 'hard_job_memory_limit_bytes': None}

    def terminate(self, code=42):
        self._check(self.api.TerminateJobObject(self._open(), code))

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            self._check(self.api.CloseHandle(handle))
