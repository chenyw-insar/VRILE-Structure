"""Bounded cleanup of one renderer session; contains no rendering/science logic."""
import os
import signal
import subprocess
import time


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def terminate_process_group(child, term_timeout=3.0, kill_timeout=3.0):
    """TERM, bounded wait, KILL the remaining group, then reap the direct child.

Check the process group, not just Popen.poll(): a leader can exit on TERM while
its GMT/shell descendant ignores TERM and keeps the group alive.
"""
    pgid = child.pid  # Caller starts a new session for this child.
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + term_timeout
    while group_exists(pgid) and time.monotonic() < deadline:
        child.poll()  # Reap an exited leader without assuming descendants exited.
        time.sleep(0.025)
    if group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=kill_timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('RENDERER_PROCESS_REAP_FAILED') from exc
