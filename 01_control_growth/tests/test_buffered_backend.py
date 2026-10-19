import os
import sys

import emcee
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffered_backend import BufferedHDFBackend  # noqa: E402
from models import _closed_form, solve_model  # noqa: E402


def _log_prob(x):
    return -0.5 * np.sum(x ** 2)


def _run(backend, steps, start=None, seed=3):
    sampler = emcee.EnsembleSampler(6, 2, _log_prob, backend=backend)
    if start is not None:
        np.random.seed(seed)
        sampler._random.seed(seed)
    sampler.run_mcmc(start, steps)
    if isinstance(backend, BufferedHDFBackend):
        backend.flush()


def test_buffered_backend_matches_standard_backend(tmp_path):
    """Standard run of 60 steps vs 20 standard steps resumed with 40 buffered steps (blocks of 7)."""
    start = np.random.default_rng(0).normal(size=(6, 2))
    ref = emcee.backends.HDFBackend(str(tmp_path / 'ref.h5'))
    ref.reset(6, 2)
    _run(ref, 60, start)

    first = emcee.backends.HDFBackend(str(tmp_path / 'buf.h5'))
    first.reset(6, 2)
    _run(first, 20, start)
    buffered = BufferedHDFBackend(str(tmp_path / 'buf.h5'), flush_steps=7)
    _run(buffered, 40)

    check = emcee.backends.HDFBackend(str(tmp_path / 'buf.h5'), read_only=True)
    assert check.iteration == ref.iteration == 60
    assert np.array_equal(check.get_chain(), ref.get_chain())
    assert np.array_equal(check.get_log_prob(), ref.get_log_prob())
    assert np.array_equal(check.accepted, ref.accepted)


def test_closed_form_logistic_matches_odeint():
    t = np.linspace(0.0, 40.0, 21)
    exact = _closed_form('_log', t, [120.0], (0.21, 2500.0))[:, 0]
    numeric = solve_model('_log', t, (0.21, 2500.0), 120.0)[:, 0]
    assert np.allclose(exact, numeric, rtol=1e-6)
    assert np.allclose(_closed_form('_exp', t, [50.0], (0.1,))[:, 0], 50.0 * np.exp(0.1 * t))
