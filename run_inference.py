import os
import sys
import sys
from unittest.mock import MagicMock
sys.modules["pyopenjtalk"] = MagicMock()
sys.modules["inflect"] = MagicMock()
sys.modules["unidecode"] = MagicMock()
sys.modules["eng_to_ipa"] = MagicMock()
sys.modules["pypinyin"] = MagicMock()
sys.modules["jieba"] = MagicMock()

from text.khmer import khmer_g2p

def main():
    text = "នៅថ្ងៃទី១២ ខែមីនា ឆ្នាំ២០២៥ ម៉ោង ៧:៣០ ព្រឹក ខ្ញុំបានទៅផ្សារទិញស្ករ ត្នោត ៣ គីឡូក្រាម និងអង្ករ 5 គីឡូក្រាម ថ្លៃសរុប ១៥,០០០ ៛ និង 20 ដុល្លារ។ ក្រោយមកអ្នកលក់បានបញ្ចុះតម្លៃ ១០% ដូច្នេះខ្ញុំបានបង់ត្រឹមតែ ១៣,៥០០ ៛ និង 9.99"
    
    print(f"Input text: {text}")
    
    output_words = khmer_g2p(text, auto_segment=True, return_word_list=True)
    
    
    converted_text = " | ".join(output_words)
        
    print("-" * 20)
    print(f"Output (Word-Separated): {converted_text}")
    print("-" * 20)
    
    output_file = 'output_converted.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(converted_text)
    
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
