import hashlib
import json
import os
import re
import sys
import subprocess
import tempfile

import torchaudio
import librosa
import torch


def normalize_text(text: str) -> str:
    unicode_conversion = {
        8175: "'",
        8189: "'",
        8190: "'",
        8208: "-",
        8209: "-",
        8210: "-",
        8211: "-",
        8212: "-",
        8213: "-",
        8214: "||",
        8216: "'",
        8217: "'",
        8218: ",",
        8219: "`",
        8220: '"',
        8221: '"',
        8222: ",,",
        8223: '"',
        8228: ".",
        8229: "..",
        8230: "...",
        8242: "'",
        8243: '"',
        8245: "'",
        8246: '"',
        180: "'",
        2122: "TM",  # Trademark
    }

    text = text.translate(unicode_conversion)

    non_bpe_chars = set([c for c in list(text) if ord(c) >= 256])
    if len(non_bpe_chars) > 0:
        non_bpe_points = [(c, ord(c)) for c in non_bpe_chars]
        raise ValueError(f"Non-supported character found: {non_bpe_points}")

    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ").replace("*", " ").strip()
    text = re.sub("\s\s+", " ", text)  # remove multiple spaces
    return text


def check_audio_file(path_or_uri, threshold_s=30):
    if "http" in path_or_uri:
        temp_fd, filepath = tempfile.mkstemp()
        os.close(temp_fd)  # Close the file descriptor, curl will create a new connection
        curl_command = ["curl", "-L", path_or_uri, "-o", filepath]
        subprocess.run(curl_command, check=True)

    else:
        filepath = path_or_uri

    audio, sr = librosa.load(filepath)
    duration_s = librosa.get_duration(y=audio, sr=sr)
    if duration_s < threshold_s:
        raise Exception(
            f"The audio file is too short. Please provide an audio file that is at least {threshold_s} seconds long to proceed."
        )

    # Clean up the temporary file if it was created
    if "http" in path_or_uri:
        os.remove(filepath)


def get_default_dtype() -> str:
    """Compute default 'dtype' based on GPU architecture"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_properties = torch.cuda.get_device_properties(i)
            dtype = "float16" if device_properties.major <= 7 else "bfloat16"  # tesla and turing architectures
    else:
        dtype = "float16"

    print(f"using dtype={dtype}")
    return dtype


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def hash_dictionary(d: dict):
    # Serialize the dictionary into JSON with sorted keys to ensure consistency
    serialized = json.dumps(d, sort_keys=True)
    # Encode the serialized string to bytes
    encoded = serialized.encode()
    # Create a hash object (you can also use sha1, sha512, etc.)
    hash_object = hashlib.sha256(encoded)
    # Get the hexadecimal digest of the hash
    hash_digest = hash_object.hexdigest()
    return hash_digest


# https://github.com/facebookresearch/audiocraft/blob/72cb16f9fb239e9cf03f7bd997198c7d7a67a01c/audiocraft/data/audio_utils.py#L86
def _clip_wav(wav: torch.Tensor, log_clipping: bool = False, stem_name: str | None = None) -> None:
    """Utility function to clip the audio with logging if specified."""
    max_scale = wav.abs().max()
    if log_clipping and max_scale > 1:
        clamp_prob = (wav.abs() > 1).float().mean().item()
        print(f"CLIPPING {stem_name or ''} happening with proba (a bit of clipping is okay):",
              clamp_prob, "maximum scale: ", max_scale.item(), file=sys.stderr)
    wav.clamp_(-1, 1)


# https://github.com/facebookresearch/audiocraft/blob/72cb16f9fb239e9cf03f7bd997198c7d7a67a01c/audiocraft/data/audio_utils.py#L57
def normalize_loudness(wav: torch.Tensor, sample_rate: int, loudness_headroom_db: float = 14,
                       loudness_compressor: bool = False, energy_floor: float = 2e-3):
    """Normalize an input signal to a user loudness in dB LKFS.
    Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.

    Args:
        wav (torch.Tensor): Input multichannel audio data.
        sample_rate (int): Sample rate.
        loudness_headroom_db (float): Target loudness of the output in dB LUFS.
        loudness_compressor (bool): Uses tanh for soft clipping.
        energy_floor (float): anything below that RMS level will not be rescaled.
    Returns:
        torch.Tensor: Loudness normalized output data.
    """
    energy = wav.pow(2).mean().sqrt().item()
    if energy < energy_floor:
        return wav
    transform = torchaudio.transforms.Loudness(sample_rate)
    input_loudness_db = transform(wav).item()
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = -loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    output = gain * wav
    if loudness_compressor:
        output = torch.tanh(output)
    assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
    return output


# https://github.com/facebookresearch/audiocraft/blob/72cb16f9fb239e9cf03f7bd997198c7d7a67a01c/audiocraft/data/audio_utils.py#L96
def normalize_audio(wav: torch.Tensor, normalize: bool = True,
                    strategy: str = 'peak', peak_clip_headroom_db: float = 1,
                    rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
                    loudness_compressor: bool = False, log_clipping: bool = False,
                    sample_rate: int | None = None,
                    stem_name: str | None = None) -> torch.Tensor:
    """Normalize the audio according to the prescribed strategy (see after).

    Args:
        wav (torch.Tensor): Audio data.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        loudness_compressor (bool): If True, uses tanh based soft clipping.
        log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        sample_rate (int): Sample rate for the audio data (required for loudness).
        stem_name (str, optional): Stem name for clipping logging.
    Returns:
        torch.Tensor: Normalized audio.
    """
    scale_peak = 10 ** (-peak_clip_headroom_db / 20)
    scale_rms = 10 ** (-rms_headroom_db / 20)
    if strategy == 'peak':
        rescaling = (scale_peak / wav.abs().max())
        if normalize or rescaling < 1:
            wav = wav * rescaling
    elif strategy == 'clip':
        wav = wav.clamp(-scale_peak, scale_peak)
    elif strategy == 'rms':
        mono = wav.mean(dim=0)
        rescaling = scale_rms / mono.pow(2).mean().sqrt()
        if normalize or rescaling < 1:
            wav = wav * rescaling
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    elif strategy == 'loudness':
        assert sample_rate is not None, "Loudness normalization requires sample rate."
        wav = normalize_loudness(wav, sample_rate, loudness_headroom_db, loudness_compressor)
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    else:
        assert wav.abs().max() < 1
        assert strategy == '' or strategy == 'none', f"Unexpected strategy: '{strategy}'"
    return wav
