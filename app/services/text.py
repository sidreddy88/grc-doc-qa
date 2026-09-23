import re

_WHITESPACE = re.compile(r"\s+")
_QUOTE_FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


# Keeps compound identifiers intact: "cc2.3", "aes-256", "tls1.2".
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be been by can do does for from has have how i if in is it its of on or our "
    "that the their them there these this those to was we were what when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(fold_for_match(text)) if token not in _STOPWORDS]


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def fold_for_match(text: str) -> str:
    """Lowercase and unify typographic quotes/dashes without changing string length.

    Length is preserved so match offsets in the folded string index the original text.
    """
    folded = text.translate(_QUOTE_FOLD)
    return "".join(char.lower() if len(char.lower()) == 1 else char for char in folded)
