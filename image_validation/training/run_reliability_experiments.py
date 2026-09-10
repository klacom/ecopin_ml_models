"""
run_reliability_experiments.py -- Sequential runner for Experiments 9-14

Multi-seed reliability protocol:
  Config A: H-flip p=0.2, NO rotation
  Config B: H-flip p=0.2, RandomRotation(15deg)

Runs:
  Exp9  = Seed 0,    Config A
  Exp10 = Seed 0,    Config B
  Exp11 = Seed 7,    Config A
  Exp12 = Seed 7,    Config B
  Exp13 = Seed 2024, Config A
  Exp14 = Seed 2024, Config B

Each run is fully independent with its own artifact directory.
Runs are sequential -- NOT parallel.
Test set is NOT accessed by any run.
"""

import subprocess
import sys
import os

EXPERIMENTS = [
    ('exp9',  0,    'A'),
    ('exp10', 0,    'B'),
    ('exp11', 7,    'A'),
    ('exp12', 7,    'B'),
    ('exp13', 2024, 'A'),
    ('exp14', 2024, 'B'),
]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE_SCRIPT  = os.path.join(PROJECT_ROOT, 'training', 'train_reliability_base.py')


def main():
    total = len(EXPERIMENTS)
    print(f'=== Multi-Seed Reliability Runner: {total} experiments ===')
    print('Config A: H-flip 0.2, No Rotation')
    print('Config B: H-flip 0.2, Rotation 15deg')
    print('=' * 55)

    results = []
    for i, (exp_id, seed, config) in enumerate(EXPERIMENTS, 1):
        print(f'\n[{i}/{total}] Starting {exp_id.upper()} (seed={seed}, config={config}) ...')
        cmd = [sys.executable, '-u', BASE_SCRIPT,
               '--exp_id', exp_id,
               '--seed',   str(seed),
               '--config', config]
        ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
        status = 'OK' if ret.returncode == 0 else f'FAILED (code {ret.returncode})'
        results.append((exp_id, seed, config, status))
        print(f'[{i}/{total}] {exp_id.upper()} -> {status}')

    print('\n=== Run Summary ===')
    for exp_id, seed, config, status in results:
        print(f'  {exp_id:6s}  seed={seed:4d}  config={config}  {status}')
    failed = [r for r in results if r[3] != 'OK']
    if failed:
        print(f'\nWARNING: {len(failed)} run(s) failed.')
        sys.exit(1)
    else:
        print('\nAll 6 runs completed successfully.')


if __name__ == '__main__':
    main()
