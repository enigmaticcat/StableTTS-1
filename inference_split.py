import os
import sys
from unittest.mock import MagicMock

# Mock dependencies to avoid import errors
sys.modules["pyopenjtalk"] = MagicMock()
sys.modules["inflect"] = MagicMock()
sys.modules["unidecode"] = MagicMock()
sys.modules["eng_to_ipa"] = MagicMock()
sys.modules["pypinyin"] = MagicMock()
sys.modules["jieba"] = MagicMock()

from text.khmer import khmer_g2p
from text.khmer_textnorm import normalize_khmer_text
from text.khmer_word_segmentation import segment_khmer_text

def step_1_normalize_and_romanize(text: str):
    print("\n--- STEP 1: NORMALIZE + SEGMENT ---")
    normalized = normalize_khmer_text(text)
    segmented = segment_khmer_text(normalized, method="bidirectional")
    print(f"Normalized Khmer: {normalized}")
    print(f"Segmented Khmer:  {segmented}")

    print("\n--- STEP 2: ROMANIZATION (G2P) ---")
    phonemes = khmer_g2p(text, auto_segment=True)
    print(f"Phonemes (romanized): {phonemes}")
    return segmented, phonemes

def main():
    text = "នៅថ្ងៃទី១២ ខែមីនា ឆ្នាំ២០២៥ ម៉ោង ៧:៣០ ព្រឹក ខ្ញុំបានទៅផ្សារទិញស្ករ ត្នោត ៣ គីឡូក្រាម និងអង្ករ 5 គីឡូក្រាម ថ្លៃសរុប ១៥,០០០ ៛ និង 20 ដុល្លារ។ ក្រោយមកអ្នកលក់បានបញ្ចុះតម្លៃ ១០% ដូច្នេះខ្ញុំបានបង់ត្រឹមតែ ១៣,៥០០ ៛ និង 9.99" 
    segmented, phonemes = step_1_normalize_and_romanize(text)


if __name__ == "__main__":
    main()
