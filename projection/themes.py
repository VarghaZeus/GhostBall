"""Overlay palettes.

A theme is the whole visual identity of the projection in one object: line
colours, text colour, dash rhythm, and two behaviour switches (``glow``,
``minimal_ui``). Everything that draws takes a :class:`Theme` rather than
reaching into :mod:`app.config`, so adding a palette is a dict entry here and
touches no drawing code.

Colours are **RGB, 0-255**, matching ``config.yaml``. The animation spec wrote
its palette as OpenCV ``Scalar`` literals, which are BGR, and consequently
labelled several of them with the wrong colour name -- ``(0, 255, 255)`` is
described there as cyan, which it is in RGB and is not in BGR. RGB throughout is
the fix; :mod:`projection.draw` swaps to BGR at the single point it matters.

Choosing colours for a projector on green felt
----------------------------------------------
Two constraints that do not apply to a screen palette, and that every theme here
respects:

**No dark colours.** The projector adds light and cannot subtract it, so an RGB
near ``(0,0,0)`` is not "dark grey", it is *nothing* -- the felt shows through
unchanged. There is no such thing as a dark theme in the usual sense; the
``dark_mode`` preset is a *dimmer* theme, achieved with low alpha rather than
dark colour.

**No saturated green.** Green marks on green cloth have poor contrast at any
brightness, because the felt is already returning that wavelength. Cyan, white,
magenta, amber and mint read far better. ``classic`` is nominally green and is
deliberately pushed toward mint for exactly this reason.

Which theme is configurable
---------------------------
``classic`` takes its colours from ``render.*_color`` in ``config.yaml``, so the
documented YAML fields stay meaningful and a table with unusual felt can be
tuned without editing Python. The other presets are constants -- the point of
picking ``neon`` is to get neon, not to get whatever the YAML happens to say.
:func:`resolve_theme` is the only function that knows this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from app.config import Settings, get_settings
from app.models import Ball, BallColor, BallKind

logger = logging.getLogger(__name__)

RGB = tuple[int, int, int]

__all__ = [
    "RGB",
    "Theme",
    "THEMES",
    "DEFAULT_THEME",
    "BALL_DISPLAY_RGB",
    "get_theme",
    "resolve_theme",
    "theme_names",
    "ball_display_color",
    "mix",
    "dim",
]


@dataclass(frozen=True, slots=True)
class Theme:
    """One overlay palette.

    Frozen so a theme cannot be mutated by the layer that borrowed it -- the
    renderer holds one for the length of a frame and the effect system holds the
    same object, and a stray in-place tweak would show up as a colour that
    changes depending on draw order.
    """

    name: str

    #: Cue-ball predicted path. The brightest thing on the table: it is the one
    #: mark the player is actually aiming with.
    cue_path: RGB
    #: Object-ball post-impact paths. Deliberately dimmer than ``cue_path`` so
    #: the primary aiming line stays readable in a crowded prediction.
    object_path: RGB
    #: Impact markers and cut-angle indicators.
    impact: RGB
    #: Ghost-ball outlines at predicted resting positions.
    ghost_ball: RGB
    #: Ring drawn around a pocket a ball is predicted into.
    pocket_highlight: RGB
    #: All text.
    text: RGB
    #: Motion trail behind a moving ball.
    trail: RGB
    #: Score popups, combo badges, turn indicator -- the celebratory colour.
    accent: RGB
    #: Timer under ten seconds, fouls, "missed by" feedback.
    alert: RGB

    #: Dash and gap length in projector px for animated paths. A longer dash
    #: reads as a more solid line; a longer gap shows more felt. ``pro`` uses a
    #: near-solid rhythm, ``neon`` a sparser one.
    dash_px: int = 22
    gap_px: int = 14
    #: Px/sec the dash pattern crawls along the path. The motion is what makes a
    #: dashed line read as *directional* rather than as a row of ticks.
    dash_speed_px_s: float = 90.0

    #: Draw a wide dim halo under every line. Costs two extra polyline passes
    #: per path -- worth it on a bright projector, and the first thing to turn
    #: off if the render stage is over budget.
    glow: bool = False
    #: Suppress decorative UI (mode badge, per-impact angle text), keeping score
    #: and turn only. For players who find the full overlay busy.
    minimal_ui: bool = False

    #: Multiplier on the *rendered* alpha of secondary marks, on top of the
    #: global ``projector.overlay_alpha_pct``. Lets a theme be quieter overall
    #: without the user dimming the whole projection.
    secondary_alpha: float = 0.75


#: ``classic`` -- the default. Mint rather than green, per the module note.
#: These values duplicate the ``render.*_color`` defaults in :mod:`app.config`;
#: :func:`resolve_theme` overrides them from live settings, so they are only what
#: you get when settings are unavailable.
CLASSIC = Theme(
    name="classic",
    cue_path=(80, 255, 120),
    object_path=(150, 220, 255),
    impact=(255, 220, 80),
    ghost_ball=(220, 220, 220),
    pocket_highlight=(255, 120, 200),
    text=(255, 255, 255),
    trail=(120, 255, 180),
    accent=(120, 255, 160),
    alert=(255, 110, 90),
)

#: ``neon`` -- cyan and magenta, glowing. The most visible theme in a bright
#: room and the most expensive to render.
NEON = Theme(
    name="neon",
    cue_path=(0, 255, 255),
    object_path=(255, 90, 255),
    impact=(255, 255, 120),
    ghost_ball=(180, 240, 255),
    pocket_highlight=(255, 0, 200),
    text=(180, 255, 255),
    trail=(255, 120, 255),
    accent=(255, 0, 220),
    alert=(255, 80, 120),
    dash_px=16,
    gap_px=18,
    dash_speed_px_s=130.0,
    glow=True,
    secondary_alpha=0.8,
)

#: ``dark_mode`` -- a *dimmer* theme, not a dark one. Cool blues at reduced
#: alpha, for a room where the full-brightness overlay is distracting.
DARK_MODE = Theme(
    name="dark_mode",
    cue_path=(140, 200, 255),
    object_path=(90, 150, 210),
    impact=(200, 220, 255),
    ghost_ball=(170, 180, 190),
    pocket_highlight=(120, 255, 210),
    text=(210, 220, 230),
    trail=(90, 160, 220),
    accent=(120, 255, 210),
    alert=(255, 150, 130),
    dash_px=26,
    gap_px=10,
    dash_speed_px_s=60.0,
    secondary_alpha=0.55,
)

#: ``pro`` -- amber, near-solid lines, minimal UI. For a player who wants the
#: aiming line and nothing else.
PRO = Theme(
    name="pro",
    cue_path=(255, 205, 60),
    object_path=(210, 180, 130),
    impact=(255, 245, 200),
    ghost_ball=(235, 225, 200),
    pocket_highlight=(255, 235, 150),
    text=(255, 255, 255),
    trail=(220, 180, 80),
    accent=(255, 215, 90),
    alert=(255, 120, 80),
    dash_px=40,
    gap_px=8,
    dash_speed_px_s=45.0,
    minimal_ui=True,
    secondary_alpha=0.6,
)

THEMES: dict[str, Theme] = {theme.name: theme for theme in (CLASSIC, NEON, DARK_MODE, PRO)}

DEFAULT_THEME = CLASSIC.name

#: Ball colours as they should be *projected*, which is not the same as the ball
#: colour as detected.
#:
#: Two departures, both forced by the medium. The 8 ball cannot be drawn in
#: black -- black is transparent -- so it gets violet, the nearest hue that
#: reads as "not one of the numbered groups". And every value is lifted well
#: clear of zero: a tint at 40% brightness is invisible on felt once the global
#: overlay alpha has been applied on top of it.
BALL_DISPLAY_RGB: dict[BallColor, RGB] = {
    BallColor.WHITE: (255, 255, 255),
    BallColor.YELLOW: (255, 225, 90),
    BallColor.BLUE: (110, 170, 255),
    BallColor.RED: (255, 120, 110),
    BallColor.PURPLE: (200, 140, 255),
    BallColor.ORANGE: (255, 175, 90),
    BallColor.GREEN: (140, 255, 180),
    BallColor.MAROON: (230, 130, 150),
    BallColor.BLACK: (190, 150, 255),
    BallColor.UNKNOWN: (215, 215, 215),
}

#: Names already warned about, so an unknown theme does not log 30 times a
#: second.
_warned_names: set[str] = set()


def theme_names() -> list[str]:
    """Every registered theme name, sorted. Used to validate API input."""
    return sorted(THEMES)


def get_theme(name: str) -> Theme:
    """Look up a theme by name, falling back to ``classic``.

    Never raises. An unknown name reaches here from a hand-edited config or an
    API caller, and refusing to render is a worse answer than rendering in the
    default palette and saying so in the log.
    """
    theme = THEMES.get(name)
    if theme is None:
        if name not in _warned_names:
            _warned_names.add(name)
            logger.warning(
                "unknown theme %r; using %s (available: %s)",
                name,
                DEFAULT_THEME,
                ", ".join(theme_names()),
            )
        return THEMES[DEFAULT_THEME]
    return theme


def resolve_theme(settings: Settings | None = None) -> Theme:
    """The active theme for the current settings.

    For ``classic``, the palette is rebuilt from the live ``render.*_color``
    values, so the YAML fields documented in ``config.yaml`` do what they say and
    a colour changed over the API takes effect on the next frame. For every
    other theme the preset wins -- see the module docstring.

    Cheap enough to call per frame: a dict lookup plus, for ``classic``, one
    frozen dataclass copy.
    """
    settings = settings or get_settings()
    theme = get_theme(settings.render.theme)
    if theme.name != CLASSIC.name:
        return theme

    render = settings.render
    return replace(
        theme,
        cue_path=tuple(render.cue_path_color),
        object_path=tuple(render.object_path_color),
        impact=tuple(render.impact_color),
        ghost_ball=tuple(render.ghost_ball_color),
        pocket_highlight=tuple(render.pocket_highlight_color),
        text=tuple(render.text_color),
    )


def palette_rgb(theme: Theme) -> list[RGB]:
    """The colours actually projected as line art, brightest first.

    Exists for :mod:`vision.detection`, which rejects its own projection when
    looking for the cue: a bright line on the felt that matches one of these is
    the overlay, not a stick. It has to follow the theme rather than read config
    directly, or switching to ``neon`` would silently reintroduce the
    overlay-detected-as-cue failure.
    """
    return [theme.cue_path, theme.object_path, theme.impact, theme.pocket_highlight, theme.trail]


# ---------------------------------------------------------------------------
# Colour arithmetic
# ---------------------------------------------------------------------------


def mix(a: RGB, b: RGB, t: float) -> RGB:
    """Linearly interpolate two colours. ``t=0`` is ``a``, ``t=1`` is ``b``."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def dim(color: RGB, factor: float) -> RGB:
    """Scale a colour's brightness, clamped to 0-255.

    Prefer adjusting the alpha channel where the intent is "fade": the display
    layer multiplies RGB by per-pixel alpha, so fading through alpha composites
    correctly against whatever is already drawn, while dimming through RGB does
    not.
    """
    return (
        min(255, max(0, int(round(color[0] * factor)))),
        min(255, max(0, int(round(color[1] * factor)))),
        min(255, max(0, int(round(color[2] * factor)))),
    )


def ball_display_color(ball: Ball, theme: Theme) -> RGB:
    """The colour to draw a ball's path and ghost outline in.

    Tinted toward the ball's own colour so a five-ball prediction is legible --
    without the tint, five overlapping paths in one palette colour cannot be told
    apart -- but mixed back toward the theme's ``object_path`` so the overlay
    still reads as one coherent design rather than a fruit salad. The cue ball is
    exempt: it always gets ``cue_path``, because it is the aiming line.

    The mix is weighted heavily toward the ball. An even blend was the first
    attempt and it failed on felt: most palette colours here are already pale,
    and averaging two pale colours of different hue lands on grey, so a
    four-ball prediction rendered as four indistinguishable grey lines. Keeping
    only a quarter of the palette colour is enough to hold the family
    resemblance while leaving each path its own hue.
    """
    if ball.kind is BallKind.CUE:
        return theme.cue_path
    tint = BALL_DISPLAY_RGB.get(ball.color, BALL_DISPLAY_RGB[BallColor.UNKNOWN])
    return mix(theme.object_path, tint, 0.75)
