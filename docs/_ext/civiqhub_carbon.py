# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Pygments dark style of the documentation site.

Carbon-only colours from the CiviqHub design system. No hue between 170 deg
and 280 deg with a saturation above 10 percent.
"""

from typing import ClassVar

from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Literal,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
    Token,
    Whitespace,
)

_TEXT = "#FAFAFA"
_MUTED = "#868686"
_NAME = "#e5e5e5"
_STRING = "#73b876"
_NUMBER = "#F2A202"
_ERROR = "#ff2f1c"


class CiviqHubCarbonDark(Style):
    """Dark highlighting on #000000, using only the carbon palette."""

    name = "civiqhub-carbon-dark"
    background_color = "#000000"
    highlight_color = "#404040"
    line_number_color = _MUTED
    line_number_background_color = "#000000"
    line_number_special_color = _TEXT
    line_number_special_background_color = "#404040"

    styles: ClassVar[dict[object, str]] = {
        Token: _TEXT,
        Text: _TEXT,
        Whitespace: _TEXT,
        Comment: f"italic {_MUTED}",
        Comment.Hashbang: f"italic {_MUTED}",
        Comment.Multiline: f"italic {_MUTED}",
        Comment.Preproc: f"noitalic {_MUTED}",
        Comment.PreprocFile: f"noitalic {_STRING}",
        Comment.Single: f"italic {_MUTED}",
        Comment.Special: f"italic bold {_MUTED}",
        Keyword: f"bold {_TEXT}",
        Keyword.Constant: _NUMBER,
        Keyword.Declaration: f"bold {_TEXT}",
        Keyword.Namespace: f"bold {_TEXT}",
        Keyword.Pseudo: f"bold {_TEXT}",
        Keyword.Reserved: f"bold {_TEXT}",
        Keyword.Type: f"bold {_TEXT}",
        Operator: _TEXT,
        Operator.Word: f"bold {_TEXT}",
        Punctuation: _TEXT,
        Name: _NAME,
        Name.Attribute: _NAME,
        Name.Builtin: _NAME,
        Name.Builtin.Pseudo: _NAME,
        Name.Class: _NAME,
        Name.Constant: _NUMBER,
        Name.Decorator: _NAME,
        Name.Entity: _NAME,
        Name.Exception: _NAME,
        Name.Function: _NAME,
        Name.Function.Magic: _NAME,
        Name.Label: _NAME,
        Name.Namespace: _NAME,
        Name.Other: _NAME,
        Name.Property: _NAME,
        Name.Tag: _NAME,
        Name.Variable: _NAME,
        Name.Variable.Class: _NAME,
        Name.Variable.Global: _NAME,
        Name.Variable.Instance: _NAME,
        Name.Variable.Magic: _NAME,
        Literal: _NUMBER,
        String: _STRING,
        String.Affix: _STRING,
        String.Backtick: _STRING,
        String.Char: _STRING,
        String.Delimiter: _STRING,
        String.Doc: f"italic {_STRING}",
        String.Double: _STRING,
        String.Escape: _STRING,
        String.Heredoc: _STRING,
        String.Interpol: _STRING,
        String.Other: _STRING,
        String.Regex: _STRING,
        String.Single: _STRING,
        String.Symbol: _STRING,
        Number: _NUMBER,
        Number.Bin: _NUMBER,
        Number.Float: _NUMBER,
        Number.Hex: _NUMBER,
        Number.Integer: _NUMBER,
        Number.Integer.Long: _NUMBER,
        Number.Oct: _NUMBER,
        Error: _ERROR,
        Generic.Deleted: _ERROR,
        Generic.Emph: f"italic {_TEXT}",
        Generic.Error: _ERROR,
        Generic.Heading: f"bold {_TEXT}",
        Generic.Inserted: _STRING,
        Generic.Output: _NAME,
        Generic.Prompt: f"bold {_TEXT}",
        Generic.Strong: f"bold {_TEXT}",
        Generic.Subheading: f"bold {_NAME}",
        Generic.Traceback: _ERROR,
    }
