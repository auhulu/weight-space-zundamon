"""Run an experiment with its own working directory and the shared environment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', help='Directory name under experiments/')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Python script or -m module and arguments')
    args = parser.parse_args()
    directory = (ROOT / 'experiments' / args.experiment).resolve()
    if directory.parent != ROOT / 'experiments' or not directory.is_dir():
        parser.error('Choose an existing directory directly under experiments/')
    if not args.command:
        parser.error('Specify a Python script or -m module')
    env = os.environ.copy()
    for name, relative in [('HF_HOME', 'huggingface'), ('UV_CACHE_DIR', 'uv'), ('MPLCONFIGDIR', 'matplotlib')]:
        env.setdefault(name, str(ROOT / 'cache' / relative))
    return subprocess.call([sys.executable, *args.command], cwd=directory, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
