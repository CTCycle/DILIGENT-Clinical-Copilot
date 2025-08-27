import re

# Month names in Italian
# -----------------------------------------------------------------------------
ITALIAN_MONTHS = {
    "gennaio": 1,
    "febbraio": 2,
    "marzo": 3,
    "aprile": 4,
    "maggio": 5,
    "giugno": 6,
    "luglio": 7,
    "agosto": 8,
    "settembre": 9,
    "ottobre": 10,
    "novembre": 11,
    "dicembre": 12,
}

# Recognize date markers to segment batches.
DATE_PATS = [
    re.compile(r"\((?P<d>\d{1,2}\.\d{1,2}\.\d{4})\)"),  # (30.07.2025)
    re.compile(r"(?<!\d)(?P<d>\d{1,2}\.\d{1,2}\.\d{4})(?!\d)"),  # 30.07.2025
    re.compile(  # Giugno 26.2025 / Giugno 26, 2025
        r"(?P<m>\b[A-Za-zÀ-ÿ]+)\s+(?P<day>\d{1,2})[.,]?\s*(?P<year>\d{4})",
        re.IGNORECASE,
    ),
]

# Titer-like: "ANA 1:80"
TITER_RE = re.compile(
    r"(?P<name>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-/ ]{0,40}?)\s*(?P<ratio>\d+\s*:\s*\d+)\b"
)

# Numeric with optional unit and parentheses: "ALAT 424 U/L(primo rialzo)"
NUMERIC_RE = re.compile(
    r"""
    (?P<name>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9\-/ ]{0,50}?)    # test name (lazy)
    \s*
    (?P<value>\d+(?:[.,]\d+)?)                        # number
    \s*
    (?P<unit>[A-Za-zμµ/%\.0-9\-/\^]+)?                # unit (optional)
    \s*
    (?P<paren>\([^)]+\))?                             # parentheses (optional)
    """,
    re.VERBOSE,
)

# Cutoff variants inside parentheses: (cutoff 47), (cut off 141), (cut-off 38)
CUTOFF_IN_PAREN_RE = re.compile(r"cut[\s\-]?off[: ]*\s*(\d+(?:[.,]\d+)?)", re.I)
