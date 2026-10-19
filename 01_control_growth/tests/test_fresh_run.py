"""A calibration started without a usable HDF5 backend must run.

Two defects met here. The walker initialization needs the likelihood arguments, which the sampler only
builds later; when they were not forwarded, every first run of the stage crashed while the resumed runs
kept working. And a backend that exists but holds no step -- what an interrupted first run leaves behind
-- took the resume path, so emcee was asked to continue a chain that had never started.

The first check is an actual short run, because that defect is in how `global.py` calls the
initialization, not in the initialization itself.
"""
import importlib.util
import os
import subprocess
import sys

import emcee

STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stage_module():
    """global.py cannot be imported by name, `global` being a keyword."""
    spec = importlib.util.spec_from_file_location('stage_global', os.path.join(STAGE, 'global.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_new_backend_is_initialized_around_a_map_estimate(tmp_path):
    env = dict(os.environ, CALIB_TEST='1', CALIB_USE_POOL='0')
    done = subprocess.run([sys.executable, os.path.join(STAGE, 'global.py')], cwd=tmp_path,
                          env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert '[init] ball | center log_prob' in done.stdout, done.stdout[-2000:]
    backends = list((tmp_path / 'reinitiate_files').glob('reinitiate_TEST_*.h5'))
    assert len(backends) == 1, backends


def test_an_empty_backend_is_initialized_instead_of_resumed(tmp_path):
    stage = _stage_module()
    group = 'control_resistant'
    l_bound, u_bound, _labels = stage.define_bounds_labels(
        '_log', group_size=len(stage.full_data[group]), max_ic=stage.max_ic, bool_multiple_mice=True)
    ndim = len(l_bound)
    nwalkers = 2 * ndim
    sampler_args = (l_bound, u_bound, '_log', stage.full_data, group)

    filename = str(tmp_path / 'empty.h5')
    emcee.backends.HDFBackend(filename).reset(nwalkers, ndim)   # as an interrupted first run leaves it

    backend, pos, walkers = stage._open_backend(filename, nwalkers, ndim, l_bound, u_bound,
                                                {'init': 'ball'}, sampler_args)
    assert backend.iteration == 0
    assert walkers == nwalkers
    assert pos is not None and pos.shape == (nwalkers, ndim)
