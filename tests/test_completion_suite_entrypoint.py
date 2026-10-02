from pathlib import Path
import subprocess
import sys


def test_suite_can_import_its_evidence_helper_from_an_external_working_directory(tmp_path):
    script = Path(__file__).resolve().parents[1]/'scripts/inspire/run_broader_completion_suite.py'
    # The real driver runs by absolute filename from the notebook's default cwd.
    # -I removes inherited PYTHONPATH, so this detects reliance on the test cwd.
    code = ("import runpy; runpy.run_path("+repr(str(script))+"); "
        "from scripts.reporting.collect_transition_endpoint import sha; "
        "assert len(sha("+repr(str(script))+")) == 64")
    result = subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
