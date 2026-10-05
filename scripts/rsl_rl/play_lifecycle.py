"""Bounded shutdown for standalone Linux Isaac play processes.

The helper process watches a signal wakeup pipe, so the deadline still works
when the simulator holds the GIL. It can kill only its owning play process.
"""
from __future__ import annotations

import atexit
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


def _watchdog(parent_pid: int, read_fd: int, timeout: float) -> None:
    deadline = None
    pid_fd = os.pidfd_open(parent_pid) if hasattr(os, "pidfd_open") else None
    watched = [read_fd] + ([] if pid_fd is None else [pid_fd])
    while True:
        remaining = None if deadline is None else max(0., deadline - time.monotonic())
        ready, _, _ = select.select(watched, [], [], remaining)
        if pid_fd is not None and pid_fd in ready:
            return
        if read_fd in ready:
            data = os.read(read_fd, 1024)
            if not data or b"D" in data:
                return
            if deadline is None and any(b in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, ord("Q")) for b in data):
                deadline = time.monotonic() + timeout
        if deadline is not None and time.monotonic() >= deadline:
            try:
                os.write(2, b"[play] Shutdown timed out; terminating this play process.\n")
            except OSError:
                pass  # The terminal may already have been closed.
            try:
                if pid_fd is None:
                    os.kill(parent_pid, signal.SIGKILL)
                else:
                    signal.pidfd_send_signal(pid_fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return


class PlayLifecycle:
    def __init__(self, timeout: float = 8.):
        self.app = self.env = None
        self.requested = self._closing = self._closed = False
        self._shutdown_sub = None
        self._handlers = {}
        read_fd, self._write_fd = os.pipe()
        os.set_blocking(self._write_fd, False)
        self._watcher = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--watchdog",
             str(os.getpid()), str(read_fd), str(timeout)],
            pass_fds=(read_fd,), start_new_session=True, stdin=subprocess.DEVNULL,
        )
        os.close(read_fd)
        self._previous_wakeup_fd = signal.set_wakeup_fd(self._write_fd, warn_on_full_buffer=False)
        self.install_signals()
        atexit.register(self.close)

    def install_signals(self):
        # AppLauncher installs its own handlers. Reinstall after launch too.
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self._handlers.setdefault(signum, signal.getsignal(signum))
            signal.signal(signum, self._on_signal)
        signal.set_wakeup_fd(self._write_fd, warn_on_full_buffer=False)

    def _on_signal(self, signum, _frame):
        if self.requested:
            os._exit(128 + signum)
        self.begin_shutdown()
        # A flag alone cannot escape IsaacLab's paused/stopped render loop.
        raise KeyboardInterrupt

    def begin_shutdown(self):
        self.requested = True
        try:
            os.write(self._write_fd, b"Q")
        except (OSError, BlockingIOError):
            pass

    def bind_app(self, app):
        self.app = app
        self.install_signals()
        import carb.eventdispatcher
        import omni.kit.app
        self._shutdown_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.kit.app.GLOBAL_EVENT_POST_QUIT,
            on_event=lambda event: self.begin_shutdown(),
            observer_name="play shutdown deadline", order=-100,
        )

    def bind_env(self, env):
        self.env = env
        sim = getattr(env.unwrapped, "sim", None)
        if sim is not None and hasattr(sim, "_disable_app_control_on_stop_handle"):
            # This installed IsaacLab version otherwise renders forever on STOP,
            # including during window close. Keep this change local to play.
            sim._disable_app_control_on_stop_handle = True

    def close(self):
        if self._closed or self._closing:
            return
        self._closing = True
        self.begin_shutdown()
        try:
            if self.env is not None:
                self.env.close()
        finally:
            try:
                if self.app is not None:
                    # is_running() may already be false while resources still
                    # need cleanup. SimulationApp.close is deliberately called.
                    self.app.close()
            finally:
                self._closed = True
                signal.set_wakeup_fd(self._previous_wakeup_fd)
                for signum, handler in self._handlers.items():
                    signal.signal(signum, handler)
                os.write(self._write_fd, b"D")
                os.close(self._write_fd)
                self._watcher.wait(timeout=2.)
                atexit.unregister(self.close)


if __name__ == "__main__" and sys.argv[1] == "--watchdog":
    _watchdog(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
