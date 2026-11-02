"""Double-click entry point for the portable second-reviewer Windows app."""
from pathlib import Path
import sys

from review_severity_pilot import main as run_review


def run():
    location = sys.executable if getattr(sys, 'frozen', False) else __file__
    root = Path(location).resolve().parent
    run_review(['--root', str(root), '--reviewer', 'reviewer2', *sys.argv[1:]])


if __name__ == '__main__':
    run()
