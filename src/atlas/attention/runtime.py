"""Persistent decode KV storage and numerically stable, exact partition fusion.

This is an authored dense reference runtime, not an HGCA/BEYOND reproduction.
Only unpadded, evaluation-mode float32 single-query decoding is supported.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class Plan:
    id: str
    label: str
    gpu_fraction: float
    executor: str


PLANS = {
    p.id: p for p in (
        Plan("resident", "Native GPU", 1.0, "resident"),
        Plan("gpu_dense", "GPU dense reference", 1.0, "dense_gpu"),
        Plan("copy", "Transfer to GPU", 0.0, "copy"),
        Plan("cpu", "CPU attention", 0.0, "cpu"),
        Plan("hybrid25", "Hybrid · 25% GPU", 0.25, "hybrid"),
        Plan("hybrid50", "Hybrid · 50% GPU", 0.50, "hybrid"),
        Plan("hybrid75", "Hybrid · 75% GPU", 0.75, "hybrid"),
    )
}


def partial_attention(q: Tensor, k: Tensor, v: Tensor, scale: float | None = None) -> tuple[Tensor, Tensor]:
    """Return locally normalized output and log normalizer; no empty partitions."""
    if k.shape[-2] == 0:
        raise ValueError("partial_attention requires a nonempty partition")
    scores = torch.matmul(q, k.transpose(-1, -2)) * (scale if scale is not None else q.shape[-1] ** -0.5)
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    return torch.matmul(torch.softmax(scores, dim=-1), v), lse


def fuse(a: tuple[Tensor, Tensor], b: tuple[Tensor, Tensor]) -> Tensor:
    """Exactly combine disjoint softmax partitions in real arithmetic."""
    out_a, lse_a = a
    out_b, lse_b = b
    total = torch.logaddexp(lse_a, lse_b)
    return out_a * torch.exp(lse_a - total) + out_b * torch.exp(lse_b - total)


class DecodeCache:
    """One layer's persistent, append-only old CPU / rolling recent GPU cache.

    GPU buffers are fixed capacity. CPU buffers use pinned memory on CUDA systems.
    Eviction uses explicit copies and is included when called inside token timing.
    """
    def __init__(self, k: Tensor, v: Tensor, plan: Plan, capacity: int):
        if k.shape != v.shape or k.ndim != 4 or k.dtype != torch.float32 or v.dtype!=torch.float32 or k.device!=v.device:
            raise ValueError("KV must have matching B,H,T,D float32 shapes")
        if not 0 <= plan.gpu_fraction <= 1 or capacity < k.shape[-2]:
            raise ValueError("Invalid cache capacity or partition")
        self.plan, self.device = plan, k.device
        self.capacity, self.length = capacity, k.shape[-2]
        self.window = capacity if plan.executor in ("resident", "dense_gpu") else (max(1, int(k.shape[-2] * plan.gpu_fraction)) if plan.executor == "hybrid" else 0)
        self.gpu_count = min(self.window, self.length)
        self.cpu_count = self.length - self.gpu_count
        shape = list(k.shape)
        shape[-2] = self.window
        self.gk = torch.empty(shape, device=self.device, dtype=k.dtype)
        self.gv = torch.empty_like(self.gk)
        shape[-2] = capacity if self.window < capacity else 0
        self.ck = torch.empty(shape, device="cpu", dtype=k.dtype, pin_memory=self.device.type == "cuda")
        self.cv = torch.empty(shape, device="cpu", dtype=k.dtype, pin_memory=self.device.type == "cuda")
        if self.cpu_count:
            self.ck[..., :self.cpu_count, :].copy_(k[..., :self.cpu_count, :])
            self.cv[..., :self.cpu_count, :].copy_(v[..., :self.cpu_count, :])
        if self.gpu_count:
            self.gk[..., :self.gpu_count, :].copy_(k[..., -self.gpu_count:, :])
            self.gv[..., :self.gpu_count, :].copy_(v[..., -self.gpu_count:, :])

    def append(self, k: Tensor, v: Tensor) -> None:
        if self.length >= self.capacity or k.shape[-2] != 1 or k.shape != v.shape or k.dtype!=torch.float32 or v.dtype!=torch.float32:
            raise ValueError("Cache full or append is not one token")
        if self.window == 0:
            self.ck[..., self.cpu_count:self.cpu_count+1, :].copy_(k)
            self.cv[..., self.cpu_count:self.cpu_count+1, :].copy_(v)
            self.cpu_count += 1
        else:
            if self.gpu_count == self.window:
                self.ck[..., self.cpu_count:self.cpu_count+1, :].copy_(self.gk[..., :1, :])
                self.cv[..., self.cpu_count:self.cpu_count+1, :].copy_(self.gv[..., :1, :])
                self.cpu_count += 1
                if self.window > 1:
                    self.gk[..., :-1, :].copy_(self.gk[..., 1:, :].clone())
                    self.gv[..., :-1, :].copy_(self.gv[..., 1:, :].clone())
                self.gpu_count -= 1
            self.gk[..., self.gpu_count:self.gpu_count+1, :].copy_(k)
            self.gv[..., self.gpu_count:self.gpu_count+1, :].copy_(v)
            self.gpu_count += 1
        self.length += 1

    @property
    def gpu_kv_bytes(self) -> int:
        return (self.gk.numel() + self.gv.numel()) * self.gk.element_size()

    @property
    def cpu_kv_bytes(self) -> int:
        return (self.ck.numel() + self.cv.numel()) * self.ck.element_size()

    def attend(self, q: Tensor, scale: float | None = None) -> Tensor:
        if q.shape[-2] != 1 or q.dtype != torch.float32:
            raise ValueError("Only float32 single-query decoding is supported")
        gk, gv = self.gk[..., :self.gpu_count, :], self.gv[..., :self.gpu_count, :]
        ck, cv = self.ck[..., :self.cpu_count, :], self.cv[..., :self.cpu_count, :]
        if self.plan.executor == "resident":
            return F.scaled_dot_product_attention(q, gk, gv, scale=scale)
        if self.plan.executor == "dense_gpu":
            scores=torch.matmul(q,gk.transpose(-1,-2))*(scale if scale is not None else q.shape[-1]**-.5)
            return torch.matmul(torch.softmax(scores,dim=-1),gv)
        if self.plan.executor == "copy":
            return F.scaled_dot_product_attention(q, ck.to(q.device,non_blocking=True), cv.to(q.device,non_blocking=True), scale=scale)
        cq = q.to("cpu")
        if self.plan.executor == "cpu" or not self.gpu_count:
            return partial_attention(cq, ck, cv, scale)[0].to(q.device)
        if not self.cpu_count:
            return partial_attention(q, gk, gv, scale)[0]
        # Enqueue GPU work first; dense host work runs while the GPU is executing.
        gpu = partial_attention(q, gk, gv, scale)
        cpu = partial_attention(cq, ck, cv, scale)
        return fuse(gpu, (cpu[0].to(q.device), cpu[1].to(q.device)))


def error_metrics(actual: Tensor, expected: Tensor, atol: float = 1e-4, rtol: float = 1e-4) -> dict:
    if actual.shape!=expected.shape:raise ValueError("Numerical comparison requires identical shapes")
    a, e = actual.detach().double().cpu(), expected.detach().double().cpu()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    delta = a - e
    bound=atol+rtol*e.abs()
    return {
        "passed": finite and bool(torch.all(delta.abs() <= bound)),
        "finite": finite,
        "element_count":a.numel(),
        "violating_elements":int((delta.abs()>bound).sum()) if finite else None,
        "max_normalized_error":float((delta.abs()/bound.clamp_min(1e-30)).max()) if finite else None,
        "max_abs": float(delta.abs().max()) if finite else None,
        "relative_l2": float(delta.norm() / e.norm().clamp_min(1e-30)) if finite else None,
        "atol": atol, "rtol": rtol,
    }


class CalibratedPolicy:
    """Linear interpolation of calibration medians, deterministic lexical ties."""
    version = "linear-median-v1"

    def __init__(self, rows: list[dict]):
        self.rows = [r for r in rows if r["phase"] == "calibration" and r["correctness"]["passed"]]

    @staticmethod
    def kv_mib(ident: str,length: int,heads: int,dim: int)->float:
        plan=PLANS[ident]
        multiplier=1.0 if ident in ("resident","copy") else plan.gpu_fraction
        return 2*heads*dim*length*4*multiplier/2**20

    def choose(self, length: int, heads: int, dim: int, budget_mib: float = 2048) -> dict:
        predictions = []
        for ident, plan in PLANS.items():
            xs = sorted((r for r in self.rows if r["plan"] == ident and r["heads"] == heads and r["head_dim"] == dim), key=lambda r:r["context"])
            if not xs:
                continue
            # Include a conservative transfer scratch estimate; a soft KV budget,
            # not a whole-process VRAM limit or a claim about physical residency.
            estimated_mib = self.kv_mib(ident,length,heads,dim)
            if estimated_mib > budget_mib:
                continue
            lo = max((r for r in xs if r["context"] <= length), key=lambda r:r["context"], default=xs[0])
            hi = min((r for r in xs if r["context"] >= length), key=lambda r:r["context"], default=xs[-1])
            if lo["context"] == hi["context"]:
                estimate = lo["median_ms"] * max(length / lo["context"], 0.25)
            else:
                weight = (length-lo["context"])/(hi["context"]-lo["context"])
                estimate = lo["median_ms"]*(1-weight)+hi["median_ms"]*weight
            predictions.append({"plan":ident,"predicted_ms":estimate,"estimated_kv_mib":estimated_mib})
        if not predictions:
            raise ValueError("No compatible, validated calibration fits this KV budget")
        best = min(predictions,key=lambda r:(r["predicted_ms"],r["plan"]))
        return {**best,"policy":self.version,"predictions":predictions}
