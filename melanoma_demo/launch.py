"""Create a machine-local environment on first launch, then start the demo."""
import os
import subprocess
import sys
import venv
from pathlib import Path

root = Path(__file__).resolve().parent
os.chdir(root)
if not (3, 10) <= sys.version_info[:2] <= (3, 13):
    sys.exit('Please install 64-bit Python 3.12 and launch again.')
env = root / '.demo-env'
python = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
if not python.is_file():
    venv.create(env, with_pip=True)
if not (env / '.ready').exists():
    base = [str(python), '-m', 'pip', 'install']
    torch_index = [] if sys.platform == 'darwin' else ['--index-url', 'https://download.pytorch.org/whl/cpu']
    subprocess.run(base + torch_index + ['torch==2.7.1'], check=True)
    subprocess.run(base + ['numpy==2.2.6', 'Pillow==11.3.0'], check=True)
    (env / '.ready').touch()
subprocess.run([str(python), 'demo.py'], check=True)
