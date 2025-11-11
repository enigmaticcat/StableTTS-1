import os
import re
from pathlib import Path
from dataclasses import dataclass
from tqdm.auto import tqdm


@dataclass
class DataConfig:
    dataset_path: str = './raw_datasets/khmer/wavs'
    index_path: str = './raw_datasets/khmer/line_index.tsv'
    output_filelist_path: str = './filelists/khmer.txt'


data_config = DataConfig()


def process_dataset():
    """Read the TSV index and write a filelist of `wav_path|transcript` lines.

    The TSV is expected to have two columns: filename and transcript. Columns
    may be separated by one or more tabs. Lines starting with '#' or empty
    lines are ignored.
    """
    # Ensure output dir exists
    os.makedirs(os.path.dirname(data_config.output_filelist_path), exist_ok=True)

    results = []
    dataset_path = Path(data_config.dataset_path)

    if not Path(data_config.index_path).exists():
        raise FileNotFoundError(f"Index file not found: {data_config.index_path}")

    with open(data_config.index_path, 'r', encoding='utf-8') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue

            # split on one or more tabs, maxsplit=1
            parts = re.split(r'\t+', line, maxsplit=1)
            if len(parts) == 1:
                # fallback: try splitting by whitespace
                parts = line.split(None, 1)
            if len(parts) < 2:
                print(f"Warning: cannot parse line in index: {line}")
                continue

            filename, text = parts[0].strip(), parts[1].strip()
            if not filename.lower().endswith('.wav'):
                filename = filename + '.wav'

            wav_path = dataset_path / filename
            if not wav_path.exists():
                print(f"Warning: file not found: {wav_path}")
                continue

            results.append(f"{wav_path.resolve().as_posix()}|{text}\n")

    print(f"Writing {len(results)} entries to {data_config.output_filelist_path}")
    with open(data_config.output_filelist_path, 'w', encoding='utf-8') as f:
        f.writelines(results)


if __name__ == '__main__':
    process_dataset()
