import re
import os
from pathlib import Path
from typing import List, Dict


def _default_lexicon_path() -> str:
    # default location inside the repo: text/lexicons/khmer.tsv
    return str(Path(__file__).resolve().parent.joinpath('lexicons', 'khmer.tsv'))


def load_lexicon(path: str = None) -> Dict[str, List[str]]:
    """Load a tab-separated lexicon file into a mapping.

    Expected TSV format (no header):
        orthography\tphone_sequence
    where phone_sequence is space-separated tokens, e.g.
        ខ្ញុំ	kh nhom

    Returns a dict mapping orthography -> list of phones.
    """
    lex = {}
    if path is None:
        path = _default_lexicon_path()
    try:
        p = Path(path)
        if not p.exists():
            return lex
        with p.open('r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                key = parts[0].strip()
                phones = parts[1].strip().split()
                if key:
                    lex[key] = phones
    except Exception:
        return {}
    return lex


# load lexicon at import time if present
_LEXICON = load_lexicon()


def khmer_g2p(text: str) -> str:
    """Convert Khmer text to a space-separated phoneme string."""
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    out: List[str] = []
    for token in text.split(' '):
        if token in _LEXICON:
            out.extend(_LEXICON[token])
        else:
            # fallback: split into characters
            out.extend(list(token))

    # ✅ convert list of phonemes/characters to string
    return " ".join(out)



def khmer_to_chars(text: str) -> List[str]:
    """Return list of characters (simple fallback/utility).

    Kept for backward compatibility; prefer `khmer_g2p` which uses lexicon when available.
    """
    if text is None:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    return list(text)


# export names
khmer_to_chars.__name__ = 'khmer_to_chars'
khmer_g2p.__name__ = 'khmer_g2p'
