from __future__ import annotations

import re

_EXTRA_SPACE_RE = re.compile(r"\s+")

_DIGIT_MAP = str.maketrans(
    {
        "०": "0",
        "१": "1",
        "२": "2",
        "३": "3",
        "४": "4",
        "५": "5",
        "६": "6",
        "७": "7",
        "८": "8",
        "९": "9",
        "০": "0",
        "১": "1",
        "২": "2",
        "৩": "3",
        "৪": "4",
        "৫": "5",
        "৬": "6",
        "৭": "7",
        "৮": "8",
        "৯": "9",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)


def normalize_text(text: str, force_arabic_digits: bool = True, strip_extra_whitespace: bool = True) -> str:
    normalized = text.strip()
    if force_arabic_digits:
        normalized = normalized.translate(_DIGIT_MAP)
    if strip_extra_whitespace:
        normalized = _EXTRA_SPACE_RE.sub(" ", normalized)
    return normalized
