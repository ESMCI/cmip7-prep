"""Ask the kernel to kill a pool worker when its parent dies.

A worker started by ProcessPoolExecutor keeps running when its parent is
killed rather than shut down, and goes on holding its memory until someone
finds it.  Linux's ``prctl(PR_SET_PDEATHSIG)`` has the kernel send the worker
a signal the moment its parent dies, so it cannot be left behind.

Pass ``die_with_parent`` as a pool's ``initializer`` and it runs once in every
worker. It does nothing where prctl does not exist. A pool created inside
library code takes no initializer, as the one GenTS uses does not, and its
orphans have to be found and killed by hand.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import signal
import sys

logger = logging.getLogger(__name__)

# From linux/prctl.h.
PR_SET_PDEATHSIG = 1


def die_with_parent() -> None:
    """Ask the kernel to kill this process when its parent dies.

    The signal arrives when the thread that started this process dies. For a
    pool worker that is the thread which submitted the work.
    """
    if not sys.platform.startswith("linux"):
        return

    parent = os.getppid()
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    except (OSError, AttributeError) as exc:
        logger.warning(
            "Could not ask the kernel to kill this worker with its parent, so "
            "it will have to be killed by hand if the run is killed: %s",
            exc,
        )
        return

    # The parent can die between this process starting and the request being
    # made, and no signal arrives for a death that has already happened.
    # Leaving now is what the signal would have done.
    if os.getppid() != parent:
        logger.warning("Parent already gone; exiting rather than being orphaned")
        os._exit(1)  # pylint: disable=protected-access
