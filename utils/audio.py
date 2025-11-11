import torch
from torch import Tensor
import torch.nn as nn
import torchaudio
import soundfile as sf
import os

class LinearSpectrogram(nn.Module):
    def __init__(self, n_fft, win_length, hop_length, pad, center, pad_mode):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.pad = pad
        self.center = center
        self.pad_mode = pad_mode
        
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (self.pad, self.pad), self.pad_mode).squeeze(1)
        spec = torch.stft(waveform, self.n_fft, self.hop_length, self.win_length, self.window, self.center, self.pad_mode, False, True, True)
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        return spec


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, f_min, f_max, pad, n_mels, center, pad_mode, mel_scale):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.pad = pad
        self.n_mels = n_mels
        self.center = center
        self.pad_mode = pad_mode
        self.mel_scale = mel_scale
        
        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, pad, center, pad_mode)
        self.mel_scale = torchaudio.transforms.MelScale(n_mels, sample_rate, f_min, f_max, (n_fft//2)+1, mel_scale, mel_scale)

    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def forward(self, x: Tensor) -> Tensor:
        linear_spec = self.spectrogram(x)
        x = self.mel_scale(linear_spec)
        x = self.compress(x)
        return x
    
def load_and_resample_audio(audio_path, target_sr, device='cpu'):
    if not os.path.exists(audio_path):
        print(f"[ERROR] File not found: {audio_path}")
        return None

    try:
        # ⚙️ Dùng soundfile thay vì torchaudio.load để tránh TorchCodec
        y, sr = sf.read(audio_path, always_2d=True)
        y = torch.from_numpy(y.T).float()  # [C, T]
        print(f"[INFO] Loaded audio via soundfile: {audio_path}, sr={sr}, shape={y.shape}")
    except Exception as e:
        print(f"[ERROR] Cannot load audio: {audio_path}\n{e}")
        return None

    y = y.to(device)

    # 🔉 Convert stereo → mono
    if y.size(0) > 1:
        y = y.mean(dim=0, keepdim=True)

    # 🔁 Resample nếu tần số không trùng
    if sr != target_sr:
        y = torchaudio.functional.resample(y, sr, target_sr)

    return y
