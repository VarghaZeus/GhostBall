"""The wizard's operator window: camera view, buttons, keyboard and mouse.

Phase 6.4. The calibration wizard drives two screens at once, and confusing
them is the fastest way to make the whole thing incomprehensible:

**The projector** shows marks *on the felt*. That is
:func:`projection.renderer.render_calibration_overlay` going out through
:class:`projection.display.Display`.

**The console** is what the operator looks at: the camera's view of the table
with the wizard's annotations on it, plus the buttons. That is this module.

Keeping them separate is not an implementation detail -- the user's whole task
is comparing the two, so the wizard has to be able to show them apart.

Why the console canvas is a fixed size
--------------------------------------
The window is created ``WINDOW_AUTOSIZE`` at exactly :data:`CONSOLE_WIDTH`, so
one window pixel is one canvas pixel and a mouse position needs no guessing.
The obvious alternative -- ``WINDOW_NORMAL`` with a resizable window -- reports
mouse coordinates that may or may not be scaled to the image depending on the
OpenCV build, and a corner placed 40 px out because of it is a calibration that
is quietly wrong rather than visibly broken. A fixed canvas trades a resizable
window for a coordinate mapping that is a single division.

Note on ``waitKey`` and the projector window
--------------------------------------------
HighGUI has one global key queue, so :meth:`projection.display.Display.show`'s
own ``waitKey(1)`` can consume a keystroke meant for the console. The wizard
keeps the odds low by only pushing a projector frame when the projected content
has actually changed, which leaves the console's much longer ``waitKey`` as the
overwhelmingly likely consumer. It cannot be eliminated entirely, which is the
other reason every keyboard shortcut here also exists as an on-screen button.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import Settings, get_settings
from app.models import Vec2

logger = logging.getLogger(__name__)

__all__ = ["Button", "Console", "CONSOLE_WIDTH", "MAX_BUTTONS_PER_ROW", "WINDOW_NAME"]

WINDOW_NAME = "ar_pool_calibration"

#: Console canvas width in px. 1280 fits a Pi's HDMI desktop and a laptop
#: without scrolling, and downscaling a 1920 px camera frame into it keeps the
#: overlay text -- sized as a fraction of the camera frame -- at around 48pt.
CONSOLE_WIDTH = 1280

#: Height of one row of the button strip under the camera view.
BUTTON_ROW_HEIGHT = 92

#: Buttons past this many wrap to another row. Five across 1280 px gives ~240 px
#: targets; squeezing eight onto one row would give 150 px, which is fine for a
#: mouse and marginal for a thumb -- and the busiest screen in the wizard is the
#: one the user is poking at while holding a projector with the other hand.
MAX_BUTTONS_PER_ROW = 5

_BAR_BG = (26, 22, 18)
_BUTTON_BG = (58, 52, 46)
_BUTTON_PRIMARY = (46, 120, 60)
_BUTTON_DISABLED = (38, 36, 34)
_BUTTON_EDGE = (140, 140, 140)
_LABEL = (245, 245, 245)
_LABEL_DISABLED = (110, 110, 110)
_KEYCAP = (170, 200, 170)

#: Arrow keys, as reported by ``waitKeyEx``. Two families because the codes are
#: platform-specific: Windows returns the high values, X11/GTK the low ones.
#: Both are listed rather than detected, because a wrong guess here is a fine-
#: tune screen whose arrow keys silently do nothing.
_ARROW_ACTIONS: dict[int, str] = {
    2424832: "nudge_left",
    2555904: "nudge_right",
    2490368: "nudge_up",
    2621440: "nudge_down",
    81: "nudge_left",
    83: "nudge_right",
    82: "nudge_up",
    84: "nudge_down",
}

#: Keys that mean the same thing on every screen, so no screen has to declare
#: them. ``h``/``j``/``k``/``l`` duplicate the arrows for the case where the
#: projector window has stolen focus and only ASCII gets through.
_GLOBAL_KEY_ACTIONS: dict[str, str] = {
    "\x1b": "cancel",
    "q": "cancel",
    "\r": "confirm",
    "\n": "confirm",
    " ": "confirm",
    "h": "nudge_left",
    "l": "nudge_right",
    "k": "nudge_up",
    "j": "nudge_down",
    "+": "scale_up",
    "=": "scale_up",
    "-": "scale_down",
    "[": "rotate_ccw",
    "]": "rotate_cw",
}


@dataclass(frozen=True, slots=True)
class Button:
    """One on-screen control.

    ``key`` is the keyboard equivalent and is drawn on the button, so the
    shortcut is discoverable rather than documented somewhere the user will
    never look. ``primary`` marks the button that Enter and Space activate --
    every screen should have exactly one, so a user who never touches the mouse
    can still get through the wizard.
    """

    label: str
    action: str
    key: str = ""
    enabled: bool = True
    primary: bool = False


class Console:
    """The operator window, or a scripted stand-in for it.

    Set ``headless=True`` to run without a window. Input then comes from
    ``script`` instead of from a person, which is what makes the seven screens
    testable at all -- every one of them is a loop that blocks until a human
    does something.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        headless: bool = False,
        script: list[str] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.headless = headless or script is not None
        #: Remaining scripted actions, consumed left to right by :meth:`poll`.
        self.script: list[str] = list(script or [])
        #: Last composed console image. Kept in every mode, because it is what
        #: tests assert against and what a headless run can write to disk.
        self.last_view: np.ndarray | None = None

        self._open = False
        self._pending_click: Vec2 | None = None
        self._pending_action: str | None = None
        self._buttons: list[Button] = []
        self._view_scale = 1.0
        self._view_height = 0

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> Console:
        """Create the window, or note that there is not going to be one."""
        if self.headless:
            logger.info("calibration console running headless (%d scripted actions)", len(self.script))
            self._open = True
            return self

        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        except Exception as exc:  # noqa: BLE001 - headless build, no X, no GL
            # Degrading to headless keeps the failure legible: the wizard says
            # it cannot show a window, instead of dying inside a draw call three
            # screens later.
            logger.warning("cannot open the calibration console window (%s); going headless", exc)
            self.headless = True
        self._open = True
        return self

    def close(self) -> None:
        if self._open and not self.headless:
            cv2.destroyWindow(WINDOW_NAME)
            cv2.waitKey(1)  # let the destroy actually process
        self._open = False

    def __enter__(self) -> Console:
        return self.open() if not self._open else self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- output -------------------------------------------------------------

    def show(self, camera_view_bgr: np.ndarray, buttons: list[Button] | None = None) -> np.ndarray:
        """Compose and display one console frame.

        Args:
            camera_view_bgr: The annotated camera frame, at camera resolution.
            buttons: Controls for the current screen. Remembered, so
                :meth:`poll` hit-tests against whatever was last shown rather
                than against a list the caller has to pass twice.

        Returns:
            The composed console image, so a headless caller can save it.
        """
        self._buttons = list(buttons or [])
        canvas = self._compose(camera_view_bgr, self._buttons)
        self.last_view = canvas
        if not self.headless:
            cv2.imshow(WINDOW_NAME, canvas)
        return canvas

    def _compose(self, camera_view_bgr: np.ndarray, buttons: list[Button]) -> np.ndarray:
        """Scale the camera view to the canvas and add the button strip."""
        height, width = camera_view_bgr.shape[:2]
        self._view_scale = CONSOLE_WIDTH / float(width)
        self._view_height = max(1, int(round(height * self._view_scale)))

        # INTER_AREA rather than the default: this is a downscale, and the
        # default bilinear filter aliases thin overlay lines into a dotted mess
        # -- which on a calibration tool reads as a projection problem.
        view = cv2.resize(
            camera_view_bgr, (CONSOLE_WIDTH, self._view_height), interpolation=cv2.INTER_AREA
        )

        bar = _bar_height(len(buttons))
        canvas = np.empty((self._view_height + bar, CONSOLE_WIDTH, 3), dtype=np.uint8)
        canvas[: self._view_height] = view
        canvas[self._view_height :] = _BAR_BG
        self._draw_buttons(canvas, buttons)
        return canvas

    def _draw_buttons(self, canvas: np.ndarray, buttons: list[Button]) -> None:
        """Draw the button strip. Must agree with :meth:`_button_rects`."""
        for button, (x0, y0, x1, y1) in zip(buttons, self._button_rects(len(buttons)), strict=True):
            if not button.enabled:
                fill, label = _BUTTON_DISABLED, _LABEL_DISABLED
            elif button.primary:
                fill, label = _BUTTON_PRIMARY, _LABEL
            else:
                fill, label = _BUTTON_BG, _LABEL

            cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), _BUTTON_EDGE, 1, cv2.LINE_AA)

            scale = 0.72
            text = button.label
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            # Shrink rather than clip: button widths are set by how many
            # controls a screen has, so a long label on a busy screen is normal.
            available = (x1 - x0) - 24
            if text_w > available:
                scale *= available / float(text_w)
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            cv2.putText(
                canvas,
                text,
                ((x0 + x1) // 2 - text_w // 2, (y0 + y1) // 2 + text_h // 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                label,
                2,
                cv2.LINE_AA,
            )
            if button.key:
                cap = _keycap_label(button.key)
                (cap_w, _), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.putText(
                    canvas,
                    cap,
                    ((x0 + x1) // 2 - cap_w // 2, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    _KEYCAP if button.enabled else _LABEL_DISABLED,
                    1,
                    cv2.LINE_AA,
                )

    def _button_rects(self, count: int) -> list[tuple[int, int, int, int]]:
        """Hit rectangles for ``count`` buttons, wrapped into rows.

        Equal widths rather than sized to the label. The wizard is expected to
        be poked at with a finger, and equal targets mean the smallest one is as
        large as it can be -- whereas label-sized buttons make ``[Yes]`` the
        hardest thing on screen to hit.

        Rows are balanced rather than filled: eight buttons come out 4 and 4,
        not 5 and 3. A short trailing row reads as an afterthought, and the
        controls on it (which tend to be Cancel and Back) get hit by accident.
        """
        if count <= 0:
            return []
        rows = _row_count(count)
        per_row = -(-count // rows)  # ceiling division: balances the rows
        pad = 12
        rects: list[tuple[int, int, int, int]] = []
        for index in range(count):
            row, column = divmod(index, per_row)
            in_this_row = min(per_row, count - row * per_row)
            span = (CONSOLE_WIDTH - pad * (in_this_row + 1)) / in_this_row
            left = pad + column * (span + pad)
            top = self._view_height + row * BUTTON_ROW_HEIGHT + pad
            rects.append(
                (int(left), top, int(left + span), top + BUTTON_ROW_HEIGHT - 2 * pad)
            )
        return rects

    # -- input --------------------------------------------------------------

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Window mouse callback. Runs on the GUI thread, so it only records."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if y < self._view_height:
            # Inside the camera view: hand back the point in camera px, which is
            # the space every caller works in.
            scale = self._view_scale if self._view_scale > 1e-6 else 1.0
            self._pending_click = Vec2(x / scale, y / scale)
            self._pending_action = "click"
            return
        for button, (x0, y0, x1, y1) in zip(
            self._buttons, self._button_rects(len(self._buttons)), strict=True
        ):
            if x0 <= x <= x1 and y0 <= y <= y1:
                if not button.enabled:
                    logger.debug("ignored click on disabled button %r", button.label)
                    return
                self._pending_action = button.action
                return

    def poll(self, timeout_ms: int = 25) -> str | None:
        """Wait briefly for input and return the action it maps to.

        Args:
            timeout_ms: How long to block. This is the wizard's frame pacing as
                well as its input latency -- long enough that the console is not
                spinning a core, short enough that a live error readout tracks
                the projector being pushed around.

        Returns:
            An action string, or ``None`` if nothing happened. Unrecognised keys
            return ``None`` rather than raising: a user pressing random keys
            should get nothing, not a crash.
        """
        if self.headless:
            return self._scripted_action()

        key = cv2.waitKeyEx(max(1, timeout_ms))
        action = self._pending_action
        self._pending_action = None
        if action is not None:
            logger.debug("console action %r from mouse", action)
            return action
        if key == -1:
            return None
        return self._key_action(key)

    def _key_action(self, key: int) -> str | None:
        """Map a raw key code to an action, or ``None`` if it means nothing."""
        if key in _ARROW_ACTIONS:
            return _ARROW_ACTIONS[key]

        char = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 128 else ""
        for button in self._buttons:
            if button.key and button.key.lower() == char:
                if not button.enabled:
                    logger.debug("ignored key %r for disabled button %r", char, button.label)
                    return None
                return button.action

        action = _GLOBAL_KEY_ACTIONS.get(char)
        if action == "confirm":
            # Enter and Space mean "do the obvious thing", which is whatever the
            # screen marked primary. A screen with no primary button has nothing
            # obvious to do, so they do nothing.
            primary = next((b for b in self._buttons if b.primary and b.enabled), None)
            return primary.action if primary else None
        if action is not None:
            logger.debug("console action %r from key %r", action, char)
        return action

    def _scripted_action(self) -> str | None:
        """Next action from the script, for headless runs and tests.

        An exhausted script yields ``cancel`` rather than ``None`` forever. A
        screen polls in a loop, so ``None`` would hang the test suite; ``cancel``
        unwinds the wizard the same way a user walking away would.
        """
        if not self.script:
            logger.warning("calibration script exhausted; cancelling the wizard")
            return "cancel"

        entry = self.script.pop(0)
        if entry == "wait":
            return None
        if entry.startswith("click:"):
            try:
                x_text, y_text = entry.removeprefix("click:").split(",")
                self._pending_click = Vec2(float(x_text), float(y_text))
            except ValueError:
                logger.error("malformed scripted click %r; expected 'click:x,y'", entry)
                return None
            return "click"
        return entry

    def take_click(self) -> Vec2 | None:
        """The last click inside the camera view, in camera px, consumed.

        Consumed rather than sampled, so a screen that reads it twice does not
        place two corners from one click.
        """
        point, self._pending_click = self._pending_click, None
        return point


def _row_count(button_count: int) -> int:
    """How many rows ``button_count`` buttons need."""
    return max(1, -(-button_count // MAX_BUTTONS_PER_ROW))


def _bar_height(button_count: int) -> int:
    """Height of the button strip. Always at least one row, so the console does
    not change size on a screen that happens to offer no controls."""
    return _row_count(button_count) * BUTTON_ROW_HEIGHT


def _keycap_label(key: str) -> str:
    """Printable name for a shortcut key, for drawing under a button."""
    return {" ": "SPACE", "\r": "ENTER", "\x1b": "ESC"}.get(key, key.upper())
