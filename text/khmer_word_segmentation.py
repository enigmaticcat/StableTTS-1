"""
Khmer Word Segmentation using Longest Matching Algorithm

This module provides word segmentation for Khmer text that lacks spaces
between words (e.g., output from speech recognition systems like Gemini).

Uses lexicon-based longest matching approach with multiple strategies:
- Forward longest matching (greedy from left to right)
- Reverse longest matching (greedy from right to left)
- Bidirectional matching (combines both and resolves conflicts)
"""

from pathlib import Path
from typing import List, Set, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _default_lexicon_path() -> str:
    """Get default path to Khmer lexicon file"""
    return str(Path(__file__).resolve().parent.joinpath('lexicons', 'khmer.tsv'))


def load_word_set(lexicon_path: Optional[str] = None) -> tuple[Set[str], int]:
    """Load lexicon words into a set for fast O(1) lookup.
    
    Args:
        lexicon_path: Path to TSV lexicon file (word\\tphonemes format)
                     If None, uses default path
    
    Returns:
        Tuple of (word_set, max_word_length)
    """
    word_set = set()
    max_len = 0
    
    if lexicon_path is None:
        lexicon_path = _default_lexicon_path()
    
    try:
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                word = parts[0].strip()
                if word:
                    word_set.add(word)
                    max_len = max(max_len, len(word))
        
        logger.info(f"Loaded {len(word_set)} words from lexicon, max length: {max_len}")
    except FileNotFoundError:
        logger.warning(f"Lexicon file not found: {lexicon_path}")
    except Exception as e:
        logger.error(f"Error loading lexicon: {e}")
    
    return word_set, max_len


# Global lexicon cache - load once at module import
_WORD_SET, _MAX_WORD_LEN = load_word_set()


def longest_matching(text: str, word_set: Set[str], max_word_len: int) -> List[str]:
    """Segment text using forward longest matching (greedy left-to-right).
    
    Algorithm:
    1. Start at position 0
    2. Try to match longest possible word at current position
    3. If found, add to result and advance position
    4. If not found, take single character and advance
    5. Repeat until end of text
    
    Args:
        text: Input text without spaces
        word_set: Set of valid words from lexicon
        max_word_len: Maximum word length in lexicon (for optimization)
    
    Returns:
        List of segmented words/characters
    """
    if not text:
        return []
    
    result = []
    i = 0
    text_len = len(text)
    
    while i < text_len:
        # Try to match longest word first
        matched = False
        # Limit search to max_word_len or remaining text
        max_try = min(max_word_len, text_len - i)
        
        for length in range(max_try, 0, -1):
            candidate = text[i:i+length]
            if candidate in word_set:
                result.append(candidate)
                i += length
                matched = True
                break
        
        # If no match, take single character
        if not matched:
            result.append(text[i])
            i += 1
    
    return result


def reverse_longest_matching(text: str, word_set: Set[str], max_word_len: int) -> List[str]:
    """Segment text using reverse longest matching (greedy right-to-left).
    
    Sometimes reverse matching produces better results for certain languages
    or ambiguous cases.
    
    Args:
        text: Input text without spaces
        word_set: Set of valid words from lexicon
        max_word_len: Maximum word length in lexicon
    
    Returns:
        List of segmented words/characters
    """
    if not text:
        return []
    
    result = []
    i = len(text)
    
    while i > 0:
        matched = False
        # Try to match longest word first (going backwards)
        max_try = min(max_word_len, i)
        
        for length in range(max_try, 0, -1):
            candidate = text[i-length:i]
            if candidate in word_set:
                result.insert(0, candidate)
                i -= length
                matched = True
                break
        
        # If no match, take single character
        if not matched:
            result.insert(0, text[i-1])
            i -= 1
    
    return result


def bidirectional_matching(text: str, word_set: Set[str], max_word_len: int) -> List[str]:
    """Segment using both forward and reverse matching, then resolve conflicts.
    
    Strategy:
    1. Apply both forward and reverse longest matching
    2. If results are the same, return immediately
    3. If different, use heuristics to choose better segmentation:
       - Fewer segments (fewer words) is often better
       - Longer words are preferred over shorter ones
       - If still tied, prefer forward matching
    
    Args:
        text: Input text without spaces
        word_set: Set of valid words from lexicon  
        max_word_len: Maximum word length in lexicon
    
    Returns:
        List of segmented words/characters
    """
    if not text:
        return []
    
    # Get both segmentations
    forward = longest_matching(text, word_set, max_word_len)
    reverse = reverse_longest_matching(text, word_set, max_word_len)
    
    # If they agree, use either
    if forward == reverse:
        return forward
    
    # Choose based on heuristics
    # Prefer fewer segments (longer words)
    if len(forward) < len(reverse):
        return forward
    elif len(reverse) < len(forward):
        return reverse
    
    # If same number of segments, prefer longer average word length
    forward_avg_len = sum(len(w) for w in forward) / len(forward)
    reverse_avg_len = sum(len(w) for w in reverse) / len(reverse)
    
    if forward_avg_len > reverse_avg_len:
        return forward
    elif reverse_avg_len > forward_avg_len:
        return reverse
    
    # Still tied? Default to forward
    return forward


def segment_khmer_text(
    text: str,
    method: str = 'bidirectional',
    word_set: Optional[Set[str]] = None,
    max_word_len: Optional[int] = None
) -> str:
    if not text or not text.strip():
        return text
    
    # Remove existing spaces for clean segmentation
    text = text.replace(' ', '')
    
    # Use global cache if not provided
    if word_set is None:
        word_set = _WORD_SET
    if max_word_len is None:
        max_word_len = _MAX_WORD_LEN
    
    # Check if lexicon is loaded
    if not word_set:
        logger.warning("Word set is empty, returning original text")
        return text
    
    # Apply selected method
    if method == 'forward':
        words = longest_matching(text, word_set, max_word_len)
    elif method == 'reverse':
        words = reverse_longest_matching(text, word_set, max_word_len)
    elif method == 'bidirectional':
        words = bidirectional_matching(text, word_set, max_word_len)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'forward', 'reverse', or 'bidirectional'")
    
    # Join with spaces
    return ' '.join(words)


# Convenience aliases
segment = segment_khmer_text
word_segment = segment_khmer_text
