"""Host power control -- rebooting the Pi from the control panel.

Its own module, small as it is, for two reasons.

The first is that ``sudo reboot`` is the most destructive thing in this
codebase, and the guards deciding whether to run it should be readable in one
place rather than spread through a route handler.

The second is honesty, which is the same theme as the rest of the panel. A
reboot button has a specific and very likely failure mode: the service account
has no passwordless sudoers entry, ``sudo`` needs a TTY it does not have, the
command fails instantly, and the panel reports "rebooting" to a Pi that is
still happily running. So the permission is checked *before* anything is fired,
and the check is a separate call rather than an inference.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

#: A tuple, not a shell string: no shell means nothing here can become an
#: injection if a future caller wants to pass a delay or a wall message.
#: ``sudo reboot`` rather than ``systemctl reboot`` because it is what the Pi's
#: own documentation uses, and so is what a hand-written sudoers entry is most
#: likely to already name.
REBOOT_COMMAND: tuple[str, ...] = ("sudo", "reboot")

#: ``sudo -n -l <cmd>`` answers "would this run without a password?" and does
#: nothing else. ``-n`` is the important flag: without it sudo would sit waiting
#: on a prompt no one can see.
PERMISSION_COMMAND: tuple[str, ...] = ("sudo", "-n", "-l", "reboot")

#: Attached to the refusal, because the fix is one line and being told the
#: command is far more use than being told the permission is missing.
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


def reboot_permission() -> tuple[bool, str]:
    """Whether ``sudo`` will run reboot with no password, and what it said.

    Worth a whole extra process before every reboot. Without a passwordless
    sudoers entry the real command fails in milliseconds with nothing on the
    panel to show it, which is the exact shape of lie the rest of this panel is
    built not to tell -- and it is the *default* state of a fresh service
    account, not an exotic misconfiguration.
    """
    try:
        result = subprocess.run(
            list(PERMISSION_COMMAND),
            capture_output=True,
            text=True,
            # Explicit: text=True alone decodes with the locale encoding.
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not ask sudo whether reboot is permitted: {exc}"

    if result.returncode == 0:
        return True, (result.stdout or "").strip()
    # sudo puts its refusals on stderr, and they are already written for a human
    # ("a password is required", "not allowed to execute"), so they are passed
    # through rather than replaced with a summary of themselves.
    return False, (result.stderr or result.stdout or "").strip()


def spawn_reboot() -> None:
    """Fire the reboot and return without waiting for it.

    ``Popen``, not ``run``: reboot returns once systemd has accepted the
    request, and "once" is not "immediately". Blocking uvicorn's event loop on
    it risks the process being torn down before the response it is holding ever
    reaches the phone -- which would show the operator a failed request for a
    reboot that was in fact already under way.

    Output goes nowhere on purpose. There will shortly be nothing alive to read
    it, and the permission question it might have answered was already asked by
    :func:`reboot_permission`.
    """
    logger.warning("rebooting the host: %s", " ".join(REBOOT_COMMAND))
    subprocess.Popen(
        list(REBOOT_COMMAND),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
