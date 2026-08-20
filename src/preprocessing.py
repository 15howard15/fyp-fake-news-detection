"""preprocessing.py — ONE cleaning pipeline shared by all models (Section 3.2.3)."""
import re
import html

_STOPWORDS = None


def _get_stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        import nltk
        try:
            from nltk.corpus import stopwords
            _STOPWORDS = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            from nltk.corpus import stopwords
            _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str, aggressive: bool = True, remove_stopwords: bool = True) -> str:
    """aggressive=True -> lowercase, strip URLs/HTML/punctuation, drop stopwords."""
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)

    if not aggressive:
        return _MULTISPACE_RE.sub(" ", text).strip()

    text = text.lower()
    text = _NON_ALPHANUM_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()

    if remove_stopwords:
        sw = _get_stopwords()
        text = " ".join(w for w in text.split() if w not in sw)

    return text


def clean_series(series, aggressive: bool = True, remove_stopwords: bool = True):
    """Apply clean_text to a pandas Series."""
    return series.apply(lambda t: clean_text(t, aggressive, remove_stopwords))
