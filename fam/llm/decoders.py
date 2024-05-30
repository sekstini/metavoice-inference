from datetime import datetime
import os
import pathlib
import uuid
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, cast

import julius
import torch
import torchaudio

from vocos import Vocos
from vocos.feature_extractors import EncodecFeatures

class Decoder(ABC):
    @abstractmethod
    def decode(self, tokens: list[int], ref_audio_path: Optional[str] = None, causal: Optional[bool] = None):
        raise NotImplementedError


class EncodecDecoder(Decoder):
    def __init__(
        self,
        tokeniser_decode_fn: Callable[[list[int]], str],
        data_adapter_fn: Callable[[list[list[int]]], tuple[list[int], list[list[int]]]],
        output_dir: str,
    ):
        self.sample_rate = 24_000
        self._end_of_audio_token = 1024
        self._num_codebooks = 8
        self.vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to("cuda")
        self.feature_extractor = cast(EncodecFeatures, self.vocos.feature_extractor)

        self.feature_extractor.encodec.set_target_bandwidth(6.0)
        self.bandwidth_id = torch.tensor([2], device="cuda")

        self.tokeniser_decode_fn = tokeniser_decode_fn
        self._data_adapter_fn = data_adapter_fn

        self.output_dir = pathlib.Path(output_dir).resolve()
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_audio(self, name: str, wav: torch.Tensor):
        torchaudio.save(f"{name}.wav", wav.squeeze(0).cpu(), self.sample_rate, backend="soundfile")

    def get_tokens(self, audio_path: str) -> list[list[int]]:
        """
        Utility method to get tokens from audio. Useful when you want to test reconstruction in some form (e.g.
        limited codebook reconstruction or sampling from second stage model only).
        """
        wav, sr = torchaudio.load(audio_path, backend="soundfile")
        if sr != self.sample_rate:
            wav = julius.resample_frac(wav, sr, self.sample_rate)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        wav = wav.to("cuda")
        tokens = self.feature_extractor.get_encodec_codes(wav).squeeze(0, 1)
        return tokens.tolist()

    def decode(
        self, tokens: list[list[int]], causal: bool = True, ref_audio_path: Optional[str] = None
    ) -> Union[str, torch.Tensor]:
        # TODO: this has strange behaviour -- if causal is True, it returns tokens. if causal is False, it SAVES the audio file.
        text_ids, extracted_audio_ids = self._data_adapter_fn(tokens)
        text = self.tokeniser_decode_fn(text_ids)
        # print(f"Text: {text}")

        tokens = torch.tensor(extracted_audio_ids, device="cuda").unsqueeze(0)

        if tokens.shape[1] < self._num_codebooks:
            tokens = torch.cat(
                [tokens, *[torch.ones_like(tokens[0:1, 0:1]) * 0] * (self._num_codebooks - tokens.shape[1])], dim=1
            )

        if causal:
            return tokens
        else:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                features = self.vocos.codes_to_features(tokens.transpose(0, 1))
                wav = self.vocos.decode(features, bandwidth_id=self.bandwidth_id).unsqueeze(1)
            # NOTE: we couldn't just return wav here as it goes through loudness compression etc :)

        if wav.shape[-1] < 9600:
            # this causes problem for the code below, and is also odd :)
            # first happened for tokens (1, 8, 28) -> wav (1, 1, 8960) (~320x factor in time dimension!)
            raise Exception("wav predicted is shorter than 400ms!")

        try:
            wav_file_name = self.output_dir / f"synth_{datetime.now().strftime('%y-%m-%d--%H-%M-%S')}_{text.replace(' ', '_')[:25]}_{uuid.uuid4()}"
            self._save_audio(wav_file_name, wav)
            return wav_file_name
        except Exception:
            import traceback
            print("Failed to save audio! Traceback:")
            traceback.print_exc()

            wav_file_name = self.output_dir / f"synth_{datetime.now().strftime('%y-%m-%d--%H-%M-%S')}_{uuid.uuid4()}"
            self._save_audio(wav_file_name, wav)
            return wav_file_name
