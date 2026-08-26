"""
run_lr_pilot.py -- Run Experiments 15 and 16 sequentially.
"""

import subprocess
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

def main():
    print("=== Starting Experiment 15 (Seed 42, LR_STAGE2=5e-5) ===")
    ret15 = subprocess.run([sys.executable, "-u", "training/train_experiment15.py"], cwd=PROJECT_ROOT)
    if ret15.returncode != 0:
        print("Experiment 15 failed!")
        sys.exit(1)

    print("=== Starting Experiment 16 (Seed 123, LR_STAGE2=5e-5) ===")
    ret16 = subprocess.run([sys.executable, "-u", "training/train_experiment16.py"], cwd=PROJECT_ROOT)
    if ret16.returncode != 0:
        print("Experiment 16 failed!")
        sys.exit(1)
    
    print("=== LR Pilot Completed Successfully ===")

if __name__ == '__main__':
    main()
