import os
import json
from dataclasses import asdict
from pathlib import Path

import sys
import torch
from tqdm import tqdm

# allow running this script from the recipes/ folder: add project root to sys.path
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import MelConfig
from utils.audio import LogMelSpectrogram, load_and_resample_audio
from text.khmer import khmer_g2p

# Config
mel_config = MelConfig()
output_feature_path = './stableTTS_datasets'
output_mel_dir = os.path.join(output_feature_path, 'mels')
os.makedirs(output_mel_dir, exist_ok=True)

input_filelist = './filelists/khmer.txt'
output_filelist_json = './filelists/khmer.json'

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
mel_extractor = LogMelSpectrogram(**asdict(mel_config)).to(device)

def load_filelist(path):
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('|', 1)
            if len(parts) != 2:
                continue
            audio_path, text = parts[0].strip(), parts[1].strip()
            entries.append((str(idx), audio_path, text))
    return entries


def process_all(max_items=None):
    entries = load_filelist(input_filelist)
    if max_items is not None:
        entries = entries[:max_items]
    results = []

    for idx, audio_path, text in tqdm(entries, total=len(entries)):
        # load audio (try torchaudio, fall back to soundfile)
        try:
            audio = load_and_resample_audio(audio_path, mel_config.sample_rate)
        except Exception:
            audio = None
        if audio is None:
            try:
                import soundfile as sf
                import numpy as np
                data, sr = sf.read(audio_path, dtype='float32')
                if data.ndim > 1:
                    data = data.mean(axis=1)
                data = np.expand_dims(data, 0)
                audio = torch.from_numpy(data)
                if sr != mel_config.sample_rate:
                    # simple resample via torchaudio if available
                    try:
                        audio = torch.nn.functional.interpolate(audio.unsqueeze(0), size=int(audio.size(1) * mel_config.sample_rate / sr)).squeeze(0)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Failed to load {audio_path}: {e}")
                audio = None
        if audio is None:
            print(f"Failed to load {audio_path}")
            continue

        # g2p using khmer lexicon
        phones = khmer_g2p(text)
        if len(phones) == 0:
            print(f"Warning: empty phones for {audio_path}")
            continue

        # mel
        mel = mel_extractor(audio.to(device)).cpu().squeeze(0)
        output_mel_path = os.path.join(output_mel_dir, f"{idx}_{Path(audio_path).stem}.pt")
        torch.save(mel, output_mel_path)

        results.append({'mel_path': output_mel_path, 'phone': phones, 'audio_path': audio_path, 'text': text, 'mel_length': mel.size(-1)})

    # write jsonl
    with open(output_filelist_json, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Wrote {len(results)} entries to {output_filelist_json}")


if __name__ == '__main__':
    import sys
    max_items = None
    if len(sys.argv) > 1:
        try:
            max_items = int(sys.argv[1])
        except Exception:
            max_items = None
    process_all(max_items)
