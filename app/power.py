"""Host power control -- rebooting the Pi from the control panel.

Its own module, small as it is, for two reasons.

The first is that ``sudo reboot`` is the most destructive thing in this
codebase, and the guards deciding whether to run it should be readable in one
place rather than spread through a route handler.

The second is honesty, which is the same theme as the rest of the panel. A
reboot button has a specific and very likely failure mode: the service account
has no passwordless sudoers entry, ``sudo`` needs a TTY it does not have, the
command fails instantly, and the panel reports "rebooting" to a Pi that is
still happily running. So the command is run synchronously and its own exit
status is what gets reported -- not a guess, and not a proxy question asked
beforehand. See :func:`reboot_now` for why the proxy version was worse.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

#: A tuple, not a shell string: no shell means nothing here can become an
#: injection if a future caller wants to pass a delay or a wall message.
#:
#: ``-n`` matters as much as the command. There is no terminal for sudo to
#: prompt at, so without it a missing sudoers entry makes sudo sit waiting on a
#: password nobody can type -- the request hangs until the timeout and then
#: reports something vague. With it, sudo fails in milliseconds and says
#: "a password is required", which is the one sentence that leads to the fix.
REBOOT_COMMAND: tuple[str, ...] = ("sudo", "-n", "reboot")

#: Generous. The command normally returns in milliseconds; a slow answer means
#: something unexpected is happening and waiting longer will not improve it.
REBOOT_TIMEOUT_SECONDS = 15.0

#: Attached to a permission failure, because the fix is one line and being told
#: the command is far more use than being told the permission is missing.
SUDOERS_HINT = (
    "Grant it once with a sudoers entry, e.g.: "
    "echo \"$USER ALL=(root) NOPASSWD: /sbin/reboot\" | "
    "sudo tee /etc/sudoers.d/010-ghostball-reboot"
)


def _linux() -> bool:
    """Whether this host is one ``sudo reboot`` means anything on.

    A function rather than a module constant so a test can replace it without
    mutating ``sys.platform`` for everything else in the process.
    """
    return sys.platform.startswith("linux")


def _sudo_path() -> str | None:
    """Where ``sudo`` is, if anywhere. Same reasoning as :func:`_linux`."""
    return shutil.which("sudo")


def reboot_refusal(settings) -> str | None:
    """Why this host must not be rebooted from the panel, or ``None``.

    Three guards, in order of how strong a statement they make about the machine
    the process is actually running on:

    * **Mock hardware.** A process with a synthetic camera *and* a discarded
      projector is a development process, not the rig -- and rebooting the
      laptop somebody is writing this on is a considerably worse outcome than a
      button that declines to work there. Both, not either: running headless
      over SSH with a mock projector and a real camera is a normal thing to do
      on the Pi itself.
    * **Not Linux.** ``sudo reboot`` is not a command this host has.
    * **No sudo.** It is, but it is not installed.
    """
    if settings.camera.use_mock and settings.projector.use_mock:
        return (
            "This process is running on mock hardware -- a synthetic camera and a "
            "discarded projector -- so it is not the rig. Refusing to reboot the "
            "machine you are developing on."
        )
    if not _linux():
        return (
            f"This host is {sys.platform}, and 'sudo reboot' is a Linux command. "
            "The reboot control only works on the Pi."
        )
    if _sudo_path() is None:
        return "'sudo' is not installed on this host, so the reboot cannot be issued."
    return None


def reboot_now() -> tuple[bool, str]:
    """Issue the reboot, and report what actually happened.

    The exit status *is* the answer. An earlier version of this asked a proxy
    question first -- ``sudo -n -l reboot``, "would this be permitted?" -- and
    the proxy was worse than useless: ``-l`` resolves the command against the
    caller's ``PATH``, ``/sbin`` is not always on it, and so it can answer "not
    permitted" on a host where ``sudo reboot`` works perfectly well. A button
    that declines to work on a working rig is a worse failure than no button at
    all, so the real command is now the only thing consulted.

    Synchronous on purpose. ``reboot`` returns as soon as systemd has accepted
    the request, which is well before anything starts being killed -- so there
    is normally ample time to answer the panel, and waiting means the answer can
    be the truth rather than a guess. If the process does get torn down first,
    the panel treats a connection that dies after a reboot it asked for as the
    reboot having worked, which it has.
    """
    logger.warning("issuing host reboot: %s", " ".join(REBOOT_COMMAND))
    try:
        result = subprocess.run(
            list(REBOOT_COMMAND),
            capture_output=True,
            text=True,
            # Explicit: text=True alone decodes with the locale encoding.
            encoding="utf-8",
            errors="replace",
            timeout=REBOOT_TIMEOUT_SECONDS,
            check=False,
            # No terminal to inherit, and nothing to read from one.
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, "'sudo' is not installed on this host."
    except subprocess.TimeoutExpired:
        return False, (
            f"'{' '.join(REBOOT_COMMAND)}' did not return within "
            f"{REBOOT_TIMEOUT_SECONDS:.0f}s. Either it is hung, or the host is "
            "already going down -- check whether it comes back before retrying."
        )
    except OSError as exc:
        return False, f"could not run '{' '.join(REBOOT_COMMAND)}': {exc}"

    if result.returncode == 0:
        return True, (result.stdout or "").strip()

    # sudo's own words. They are already written for a human -- "a password is
    # required", "not allowed to execute" -- and replacing them with a summary
    # of themselves would throw away the part that says what to fix.
    detail = (result.stderr or result.stdout or "").strip()
    return False, detail or f"exited {result.returncode} with no output"
