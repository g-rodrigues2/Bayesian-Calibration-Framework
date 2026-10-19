"""emcee HDF5 backend that writes the chain in blocks.

emcee.backends.HDFBackend opens the file three times per step (two consistency checks and the write).
When a step takes a few milliseconds this file access dominates the run time, especially with
antivirus scanning on Windows. BufferedHDFBackend keeps the steps in memory and appends them to the
same file layout every `flush_steps` steps or `flush_seconds` seconds, and when the run ends (call
flush() after run_mcmc, or use the backend as a context manager). A file written this way is read and
resumed by the standard HDFBackend; an interrupted run loses at most the last unflushed block.
"""
import time

import numpy as np
from emcee.backends import HDFBackend


class BufferedHDFBackend(HDFBackend):

    def __init__(self, filename, flush_steps=2000, flush_seconds=60.0, **kwargs):
        super().__init__(filename, **kwargs)
        self.flush_steps = int(flush_steps)
        self.flush_seconds = float(flush_seconds)
        self._coords, self._log_prob, self._accepted = [], [], None
        self._random_state = None
        self._last_flush = time.monotonic()
        self._shape = None

    @property
    def iteration(self):
        return super().iteration + len(self._coords)

    def grow(self, ngrow, blobs):
        self.flush()
        super().grow(ngrow, blobs)

    def save_step(self, state, accepted):
        if state.blobs is not None:
            self.flush()
            return super().save_step(state, accepted)
        if self._shape is None:
            self._shape = tuple(int(v) for v in self.shape)
        if state.coords.shape != self._shape or state.log_prob.shape != self._shape[:1]:
            raise ValueError(f'invalid state dimensions; expected {self._shape}')
        self._coords.append(np.array(state.coords, dtype=self.dtype))
        self._log_prob.append(np.array(state.log_prob, dtype=self.dtype))
        acc = np.asarray(accepted, dtype=float)
        self._accepted = acc.copy() if self._accepted is None else self._accepted + acc
        self._random_state = state.random_state
        if (len(self._coords) >= self.flush_steps
                or time.monotonic() - self._last_flush >= self.flush_seconds):
            self.flush()

    def flush(self):
        if not self._coords:
            return
        n = len(self._coords)
        with self.open('a') as f:
            g = f[self.name]
            it = int(g.attrs['iteration'])
            if g['chain'].shape[0] < it + n:
                g['chain'].resize(it + n, axis=0)
                g['log_prob'].resize(it + n, axis=0)
            g['chain'][it:it + n] = np.stack(self._coords)
            g['log_prob'][it:it + n] = np.stack(self._log_prob)
            g['accepted'][:] += self._accepted
            for i, v in enumerate(self._random_state):
                g.attrs[f'random_state_{i}'] = v
            g.attrs['iteration'] = it + n
        self._coords, self._log_prob, self._accepted = [], [], None
        self._last_flush = time.monotonic()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.flush()
        return False
