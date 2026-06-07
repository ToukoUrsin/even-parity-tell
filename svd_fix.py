"""Robust SVD shim for quimb 1.11.2 on this Apple/numba box.

Problem observed: building a CircuitMPS on the dense obfuscated peaked circuits
(48_42, 56_43, ...) triggers `divide by zero / overflow / invalid value in matmul`
warnings -> NaN/Inf flow into `svd_truncated_numba`, which then raises a *SystemError*
("CPUDispatcher(...) returned a result with an error set"). quimb's
`svd_truncated_numpy` only catches `ValueError` for its scipy fallback, so the
SystemError propagates and kills the whole build (this is why distill_cpu.py hung
then crashed, and why CircuitMPS build dies inside swap_sites_with_compress).

Fix: replace `svd_truncated_numba` with a numpy/scipy SVD that
  (a) sanitizes NaN/Inf in the input matrix to 0,
  (b) uses numpy.linalg.svd (LAPACK gesdd) with a scipy gesvd fallback,
  (c) applies the same cutoff / max_bond / absorb / renorm trimming quimb expects,
so no exception type can escape. We register it for BOTH the numba name and the
numpy-dispatch name so every code path (gate application, swap compression,
local_expectation) uses the safe version.
"""
import numpy as np


def apply():
    import quimb.tensor.decomp as D

    _trim = D._trim_and_renorm_svd_result_numba
    _numba = D.svd_truncated_numba   # the fast kernel (keep it for the hot path)

    def safe_svd_truncated(x, cutoff=-1.0, cutoff_mode=4, max_bond=-1,
                           absorb=0, renorm=0):
        a = np.asarray(x)
        finite = np.isfinite(a).all()
        if finite:
            # FAST PATH: use the original numba kernel. Catch *every* exception
            # type (numba can raise SystemError, not just ValueError) and fall
            # through to the robust path instead of crashing the whole build.
            try:
                return _numba(a, cutoff, cutoff_mode, max_bond, absorb, renorm)
            except Exception:
                pass
        else:
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        # ROBUST PATH: numpy LAPACK gesdd, then scipy gesvd as last resort.
        try:
            U, s, VH = np.linalg.svd(a, full_matrices=False)
        except Exception:
            import scipy.linalg as scla
            U, s, VH = scla.svd(a, full_matrices=False, lapack_driver="gesvd")
        return _trim(U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm)

    # Wrap the numpy-dispatch entry. svd_truncated_numpy itself calls
    # svd_truncated_numba internally, so also rebind that name to the safe one
    # (its fast path still hits the real numba kernel via the _numba closure).
    D.svd_truncated_numba = safe_svd_truncated
    try:
        D.svd_truncated.register("numpy")(safe_svd_truncated)
    except Exception:
        pass
    return "svd-safe-fast"
