"""Patches for quimb 1.14 + cupy 13 quirks that block GPU peaked-circuit runs.

1) cupy Device unhashable -> safecache (see below).
2) quimb 1.14 `apply_swap` -> `swap_sites_with_compress` leaks `swap_back` into
   the SVD backend, which doesn't accept it. Triggered on long-range gates in
   dense circuits (64_10 etc.). We shim the backend SVD entry points to drop
   any unknown kwargs."""


def apply():
    # Preferred: make Device hashable so autoray's namespace cache still works.
    try:
        from cupy.cuda.device import Device
        Device.__hash__ = lambda self: int(self.id)
        # sanity: must actually be hashable now
        hash(Device(0))
        return "device-hash"
    except Exception:
        pass
    # Fallback: replace autoray's namespace cache with a dict that coerces an
    # unhashable device (cupy.cuda.Device) in the key tuple to its repr() before
    # hashing. get_namespace reads the module global `_NAMESPACE_CACHE` at call
    # time, so swapping it here intercepts every caller regardless of how they
    # imported get_namespace. The REAL device object is still passed on to
    # AutoNamespace (only the cache key is coerced), so no functionality is lost.
    import autoray.autoray as A

    class _SafeCache(dict):
        @staticmethod
        def _coerce(key):
            try:
                hash(key)
                return key
            except TypeError:
                cls, device, dtype, submodule = key
                return (cls, repr(device), dtype, submodule)

        def __getitem__(self, key):
            return dict.__getitem__(self, self._coerce(key))

        def __setitem__(self, key, value):
            dict.__setitem__(self, self._coerce(key), value)

        def __contains__(self, key):
            return dict.__contains__(self, self._coerce(key))

    A._NAMESPACE_CACHE = _SafeCache(A._NAMESPACE_CACHE)
    _patch_svd_drop_unknown_kwargs()
    return "safecache+svdshim"


_BAD_COMPRESS_KEYS = ("swap_back",)


def _patch_svd_drop_unknown_kwargs():
    """quimb 1.14: apply_swap -> swap_sites_with_compress builds compress_opts
    that include `swap_back=True`. Those opts get **kwargs-piped through
    tensor_split -> array_split -> backend SVD, which doesn't accept the kwarg.
    We strip the bad keys at every layer that pipes **kwargs downstream."""
    import quimb.tensor.tensor_core as TC

    def _wrap(fn):
        if getattr(fn, "__wrapped_swap_back__", False):
            return fn

        def wrapper(*a, **kw):
            for k in _BAD_COMPRESS_KEYS:
                kw.pop(k, None)
            return fn(*a, **kw)

        wrapper.__wrapped_swap_back__ = True
        wrapper.__name__ = getattr(fn, "__name__", "wrapped")
        return wrapper

    for name in ("tensor_split",):
        if hasattr(TC, name):
            setattr(TC, name, _wrap(getattr(TC, name)))
    # tensor_core.Tensor.split calls module-level tensor_split, so the wrap
    # above is sufficient. Also wrap array_split as a belt-and-braces guard.
    try:
        import quimb.tensor.decomp as D
        if hasattr(D, "array_split"):
            D.array_split = _wrap(D.array_split)
    except Exception:
        pass
