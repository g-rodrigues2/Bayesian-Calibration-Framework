"""Progress of the emcee HDF5 backends of a stage (number of steps against a target).

    python backend_progress.py [reinitiate_files] --target 30000

Backends with the TEST_ prefix are ignored. Resubmitting the same job completes partial chains, because
global.py resumes each backend up to CALIB_CHAIN_SIZE.
"""
import argparse
import glob
import os

import h5py


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', nargs='?', default='reinitiate_files')
    ap.add_argument('--target', type=int, default=int(os.environ.get('CALIB_CHAIN_SIZE', '30000')))
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.folder, 'reinitiate_*.h5'))
                   if not os.path.basename(f).startswith('reinitiate_TEST_'))
    done = 0
    for f in files:
        try:
            with h5py.File(f, 'r') as h:
                steps = int(h['mcmc'].attrs['iteration'])
        except Exception:
            steps = -1
        done += steps >= args.target
        status = 'complete' if steps >= args.target else 'partial'
        print(f'  {os.path.basename(f):60s} {steps:>7d}  [{status}]')
    print(f'\n{done}/{len(files)} backends reached {args.target} steps.')


if __name__ == '__main__':
    main()
