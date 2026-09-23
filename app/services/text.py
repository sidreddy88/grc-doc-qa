import re

_WHITESPACE = re.compile(r"\s+")
_QUOTE_FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def fold_for_match(text: str) -> str:
    """Lowercase and unify typographic quotes/dashes without changing string length.

    Length is preserved so match offsets in the folded string index the original text.
    """
    folded = text.translate(_QUOTE_FOLD)
    return "".join(char.lower() if len(char.lower()) == 1 else char for char in folded)
