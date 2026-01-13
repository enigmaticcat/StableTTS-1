import re
import os
from pathlib import Path
from typing import List, Dict, Union
from text.khmer_textnorm import normalize_khmer_text
from text.khmer_word_segmentation import segment_khmer_text

def _default_lexicon_path() -> str:
    # default location inside the repo: text/lexicons/khmer.tsv
    return str(Path(__file__).resolve().parent.joinpath('lexicons', 'khmer.tsv'))


def load_lexicon(path: str = None) -> Dict[str, List[str]]:
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


def khmer_g2p(text: str, auto_segment: bool = False, return_word_list: bool = False) -> Union[str, List[str]]:
    """Convert Khmer text to a space-separated phoneme string.
    
    Args:
        text: Input Khmer text
        auto_segment: If True, apply word segmentation before G2P conversion
        return_word_list: If True, return a list of strings (one string per word)
                         instead of a single flattened string.
    
    Returns:
        Space-separated phoneme/character string OR List of strings
    """
    # Apply word segmentation if requested
    text = normalize_khmer_text(text)

    if auto_segment and text:
        text = segment_khmer_text(text, method='bidirectional')
    if text is None:
        return [] if return_word_list else ""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return [] if return_word_list else ""

    words: List[str] = []
    for token in text.split(' '):
        if token in _LEXICON:
            words.append(" ".join(_LEXICON[token]))
        else:
            words.append(token)

    if return_word_list:
        return words

    # ✅ convert list of phonemes/characters to string
    return " ".join(words)



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
