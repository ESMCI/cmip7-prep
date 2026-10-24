"""Tests for asking the kernel to kill a pool worker with its parent."""

import ctypes
import os
import signal
from dataclasses import dataclass, field

import pytest

from cmip7_prep import worker_lifetime
from cmip7_prep.worker_lifetime import PR_SET_PDEATHSIG, die_with_parent


@dataclass
class FakeLibc:
    """A libc whose prctl records its arguments and returns what it is told."""

    result: int = 0
    calls: list = field(default_factory=list)

    def prctl(self, *args):
        """Record the call and return the configured result."""
        self.calls.append(args)
        return self.result


@pytest.fixture(name="libc")
def fixture_libc(monkeypatch):
    """Stand a fake libc in for the real one, on a platform reported as Linux."""
    fake = FakeLibc()
    monkeypatch.setattr(worker_lifetime.sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: fake)
    return fake


class TestDieWithParent:
    """What the request does, and what it does when it cannot be made."""

    def test_asks_for_sigkill_on_parent_death(self, libc):
        """prctl is called with PR_SET_PDEATHSIG and SIGKILL."""
        die_with_parent()
        assert libc.calls == [(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)]

    def test_does_nothing_off_linux(self, monkeypatch):
        """prctl does not exist there, so nothing is attempted."""
        monkeypatch.setattr(worker_lifetime.sys, "platform", "darwin")

        def refuse(*_args, **_kwargs):
            raise AssertionError("libc should not be loaded off Linux")

        monkeypatch.setattr(ctypes, "CDLL", refuse)
        die_with_parent()

    def test_a_refused_request_warns_rather_than_raising(self, libc, caplog):
        """The worker still runs; it just has to be killed by hand."""
        libc.result = -1
        die_with_parent()
        assert "killed by hand" in caplog.text

    def test_exits_when_the_parent_died_during_the_request(self, libc, monkeypatch):
        """No signal arrives for a death that already happened, so it leaves."""
        parents = iter([4242, 1])
        monkeypatch.setattr(os, "getppid", lambda: next(parents))
        leaving = []
        monkeypatch.setattr(os, "_exit", leaving.append)

        die_with_parent()

        assert libc.calls
        assert leaving == [1]

    def test_stays_when_the_parent_is_still_there(self, libc, monkeypatch):
        """The usual case: the request stands and the worker carries on."""
        monkeypatch.setattr(os, "getppid", lambda: 4242)

        def refuse(_code):
            raise AssertionError("should not exit while the parent is alive")

        monkeypatch.setattr(os, "_exit", refuse)

        die_with_parent()

        assert libc.calls
