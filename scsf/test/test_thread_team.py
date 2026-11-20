"""Stable native worker ownership and exception-barrier regression."""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_native_team_preserves_owners_and_recovers_from_worker_exception(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None: pytest.skip('C++ compiler unavailable')
    root = Path(__file__).resolve().parents[2]
    target = tmp_path / 'thread-team'
    subprocess.run([compiler, '-std=c++17', '-pthread', '-I'+str(root/'scsf/src'),
        str(root/'scsf/test/native/thread_team.cpp'), '-o', str(target)],check=True,timeout=30)
    subprocess.run([str(target)],check=True,timeout=20)
