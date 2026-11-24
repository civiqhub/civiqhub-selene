# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The colours and the ink of the figures of :mod:`civiqhub_selene.viz`.

Colour follows a fixed rule per job:

- **identity** (which group): the categorical slots of :data:`GROUP_COLORS`, assigned
  in this fixed order and never cycled; the first three slots stay apart for every
  pair of colours, under the common colour-vision deficiencies included. Past
  :data:`MAX_DISTINCT_GROUPS` groups, every figure also writes the name of each group,
  so that identity never rests on colour alone;
- **polarity** (a score from disagreement to agreement): the diverging CiviqHub Likert
  ramp of :data:`SCORE_RAMP`, two opposite hues around a neutral middle;
- **the three states of a cell**: a vote takes its score colour, a cell shown without
  a vote is drawn as a dot on the surface, a cell that is not shown is hatched.

Text is written in ink, never in the colour of a series.
"""

from typing import Final

__all__ = [
    "BLOCK_COLOR",
    "GRID",
    "GROUP_COLORS",
    "INK",
    "INK_MUTED",
    "INK_SECONDARY",
    "MAX_DISTINCT_GROUPS",
    "NOT_SHOWN",
    "SCORE_RAMP",
    "SURFACE",
    "group_color",
    "score_colors",
]

GROUP_COLORS: Final[tuple[str, ...]] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
"""Categorical colours of the groups, in their fixed order: blue, orange, aqua, yellow,
magenta, green, violet, red."""

MAX_DISTINCT_GROUPS: Final[int] = 3
"""Number of groups whose colours stay apart for every pair, colour-vision deficiencies
included."""

SCORE_RAMP: Final[tuple[str, ...]] = ("#d97858", "#ffa67c", "#e3dede", "#4794df", "#285786")
"""The CiviqHub Likert ramp for the five scores -2, -1, 0, +1, +2: orange for
disagreement, a neutral middle, blue for agreement."""

BLOCK_COLOR: Final[str] = "#52514e"
"""Blocks are drawn in ink: colour is kept for the groups."""

SURFACE: Final[str] = "#fcfcfb"
"""The surface of a figure."""

INK: Final[str] = "#0b0b0b"
"""Primary ink: titles, values, labels."""

INK_SECONDARY: Final[str] = "#52514e"
"""Secondary ink: annotations."""

INK_MUTED: Final[str] = "#898781"
"""Muted ink: axes, ticks, the dots of cells shown without a vote."""

GRID: Final[str] = "#e1e0d9"
"""Hairlines of the grid and of the separators."""

NOT_SHOWN: Final[str] = "#c3c2b7"
"""Hatching of the cells that were not shown."""


def group_color(group: int) -> str:
    """Return the colour of a group, by its index from 0.

    Groups past the eighth fold onto a neutral grey: a figure with more groups names
    them in text.
    """
    if group < 0:
        raise ValueError(f"a group index is at least 0, got {group}")
    return GROUP_COLORS[group] if group < len(GROUP_COLORS) else INK_MUTED


def score_colors(n_levels: int) -> tuple[str, ...]:
    """Return one colour per level of a scale, from the lowest score to the highest.

    With five levels, the CiviqHub Likert ramp itself; otherwise colours interpolated
    along it, so that the middle of the scale stays neutral.
    """
    if n_levels < 2:
        raise ValueError(f"a scale has at least two levels, got {n_levels}")
    if n_levels == len(SCORE_RAMP):
        return SCORE_RAMP
    anchors = [_rgb(color) for color in SCORE_RAMP]
    colors = []
    for level in range(n_levels):
        position = level / (n_levels - 1) * (len(anchors) - 1)
        low = min(int(position), len(anchors) - 2)
        weight = position - low
        mixed = tuple(
            (1 - weight) * first + weight * second
            for first, second in zip(anchors[low], anchors[low + 1], strict=True)
        )
        colors.append(_hex(mixed))
    return tuple(colors)


def _rgb(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    return (int(value[0:2], 16) / 255, int(value[2:4], 16) / 255, int(value[4:6], 16) / 255)


def _hex(rgb: tuple[float, ...]) -> str:
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)
