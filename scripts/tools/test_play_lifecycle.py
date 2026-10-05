"""Exercise real subprocess signals, including a GIL-blocking native call."""
from pathlib import Path
import select
import signal
import subprocess
import sys
import textwrap
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEADER = f"""import sys, time
sys.path.insert(0, {str(ROOT / 'scripts/rsl_rl')!r})
from play_lifecycle import PlayLifecycle
from types import SimpleNamespace
guard = PlayLifecycle(timeout=.6)
"""


def run_child(body, signum=None):
    child = subprocess.Popen([sys.executable, '-u', '-c', textwrap.dedent(HEADER) + textwrap.dedent(body)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert select.select([child.stdout], [], [], 5.)[0], 'child did not initialize'
        assert child.stdout.readline().strip() == 'READY'
        if signum is not None:
            time.sleep(.1)  # Let the native-call test enter its locked mutex.
            child.send_signal(signum)
        output, error = child.communicate(timeout=6.)
        return child.returncode, output, error
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=2.)


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_signal_closes_environment_and_app_even_when_app_not_running(signum):
    code, output, _ = run_child('''
class Env:
    unwrapped = SimpleNamespace(sim=SimpleNamespace(_disable_app_control_on_stop_handle=False))
    def close(self): print('ENV_CLOSED', flush=True)
class App:
    def is_running(self): return False
    def close(self): print('APP_CLOSED', flush=True)
env = Env()
guard.bind_env(env)
assert env.unwrapped.sim._disable_app_control_on_stop_handle
guard.app = App()
try:
    print('READY', flush=True)
    while True: time.sleep(.1)
except KeyboardInterrupt:
    pass
finally:
    guard.close()
''', signum)
    assert code == 0
    assert 'ENV_CLOSED' in output and 'APP_CLOSED' in output


def test_native_deadlock_is_killed_after_signal_without_python_handler():
    code, _, error = run_child('''
import ctypes
libc = ctypes.PyDLL(None)  # Keep the GIL during the intentionally deadlocked call.
mutex = (ctypes.c_long * 16)()
assert libc.pthread_mutex_init(ctypes.byref(mutex), None) == 0
assert libc.pthread_mutex_lock(ctypes.byref(mutex)) == 0
print('READY', flush=True)
libc.pthread_mutex_lock(ctypes.byref(mutex))
''', signal.SIGTERM)
    assert code == -signal.SIGKILL
    assert 'Shutdown timed out' in error


def test_cleanup_hang_is_bounded_without_a_signal():
    code, _, error = run_child('''
class Env:
    def close(self):
        while True: time.sleep(.1)
guard.env = Env()
print('READY', flush=True)
guard.close()
''')
    assert code == -signal.SIGKILL
    assert 'Shutdown timed out' in error
