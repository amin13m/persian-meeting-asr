from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
import librosa
import nemo.collections.asr as nemo_asr


class ShenavaKoochikEngine:
    """
    Independent inference engine for:

        Reza2kn/Shenava-Koochik-v1.0

    Backend:
        NVIDIA NeMo

    Model:
        FastConformer Hybrid RNNT/CTC

    Current decoding:
        CTC greedy_batch

    Public API:

        engine.load()

        engine.transcribe(audio)

        engine.unload()

    The engine is intentionally unaware of:
        - diarization
        - speaker isolation
        - TSE
        - ASR grouping
        - pipeline logic
        - output directories
    """

    def __init__(
        self,
        config: dict,
    ):
    
        self.config = config
    
        # ====================================================
        # Secondary ASR configuration
        # ====================================================
    
        secondary_config = config["asr"]["secondary"]
    
        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------
    
        self.model_path = Path(
            secondary_config["model"]["path"]
        )
    
        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------
    
        self.sample_rate = int(
            secondary_config.get(
                "audio",
                {},
            ).get(
                "sample_rate",
                16000,
            )
        )
    
        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------
    
        device_config = secondary_config.get(
            "device",
            {},
        )
    
        device_type = str(
            device_config.get(
                "type",
                "auto",
            )
        ).lower()
    
        self.device = self._resolve_device(
            device_type
        )
    
        # ----------------------------------------------------
        # Decoding
        # ----------------------------------------------------
    
        decoding_config = secondary_config.get(
            "decoding",
            {},
        )
    
        self.decoder = str(
            decoding_config.get(
                "decoder",
                "ctc",
            )
        ).lower()
    
        self.greedy_strategy = str(
            decoding_config.get(
                "strategy",
                "greedy_batch",
            )
        ).lower()
    
        # ----------------------------------------------------
        # Attention context
        # ----------------------------------------------------
    
        context = secondary_config.get(
            "attention_context",
            [70, 13],
        )
    
        self.attention_context = list(
            context
        )
    
        # ====================================================
        # Runtime
        # ====================================================
    
        self.model = None
        self.loaded = False

    # ========================================================
    # Device
    # ========================================================

    @staticmethod
    def _resolve_device(
        device_type: str,
    ) -> torch.device:

        if device_type == "cuda":

            if not torch.cuda.is_available():

                raise RuntimeError(
                    "CUDA was requested but "
                    "CUDA is not available."
                )

            return torch.device(
                "cuda"
            )

        if device_type == "cpu":

            return torch.device(
                "cpu"
            )

        if device_type == "auto":

            if torch.cuda.is_available():

                return torch.device(
                    "cuda"
                )

            return torch.device(
                "cpu"
            )

        raise ValueError(
            f"Unsupported device type: "
            f"{device_type}"
        )

    # ========================================================
    # Load
    # ========================================================

    def load(
        self,
    ) -> None:
    
        if self.loaded:
            return
    
        model_path = self.model_path
    
        if not model_path.is_absolute():
            model_path = (
                Path.cwd()
                / model_path
            )
    
        model_path = model_path.resolve()
    
        if not model_path.exists():
            raise FileNotFoundError(
                "Shenava model does not exist:\n"
                f"{model_path}"
            )
    
        if not model_path.is_file():
            raise ValueError(
                "Shenava model path is not a file:\n"
                f"{model_path}"
            )
    
        if model_path.suffix.lower() != ".nemo":
            raise ValueError(
                "Shenava model must be a .nemo file:\n"
                f"{model_path}"
            )
    
        self.model_path = model_path
    
        print("=" * 60)
        print("Loading Shenava Koochik")
        print(f"Model : {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Decoder: {self.decoder}")
        print(f"Strategy: {self.greedy_strategy}")
        print(f"Attention context: {self.attention_context}")
        print("=" * 60)
    
        # ----------------------------------------------------
        # Load NeMo Hybrid RNNT/CTC model
        # ----------------------------------------------------
    
        self.model = nemo_asr.models.ASRModel.restore_from(
            restore_path=str(
                self.model_path
            ),
            map_location=self.device,
        )
    
        self.model = self.model.to(
            self.device
        )
    
        self.model.eval()
    
        # ----------------------------------------------------
        # IMPORTANT:
        # Shenava Koochik is Hybrid RNNT/CTC.
        #
        # v1.0 should use the CTC head.
        # The default Hybrid decoder is RNNT.
        # ----------------------------------------------------
    
        if self.decoder == "ctc":
        
            decoding_cfg = (
                self.model.cfg.aux_ctc.decoding
            )
    
            decoding_cfg.strategy = "greedy"
    
            self.model.change_decoding_strategy(
                decoding_cfg=decoding_cfg,
                decoder_type="ctc",
                verbose=False,
            )
    
        elif self.decoder == "rnnt":
        
            decoding_cfg = (
                self.model.cfg.decoding
            )
    
            decoding_cfg.strategy = (
                self.greedy_strategy
            )
    
            self.model.change_decoding_strategy(
                decoding_cfg=decoding_cfg,
                decoder_type="rnnt",
                verbose=False,
            )
    
        else:
        
            raise ValueError(
                "Unsupported Shenava decoder: "
                f"{self.decoder}. "
                "Expected 'ctc' or 'rnnt'."
            )
    
        # ----------------------------------------------------
        # Attention context
        # ----------------------------------------------------
    
        self.model.encoder.set_default_att_context_size(
            self.attention_context
        )
    
        self.loaded = True
    
        print(
            "Shenava Koochik loaded successfully."
        )
    
        print(
            f"Active decoder: {getattr(self.model, 'cur_decoder', 'unknown')}"
        )

    # ========================================================
    # Prepare audio
    # ========================================================

    def prepare_audio(
        self,
        audio,
        sample_rate: int,
    ) -> np.ndarray:

        data = np.asarray(
            audio
        )

        if data.size == 0:

            return np.zeros(
                0,
                dtype=np.float32,
            )

        # ----------------------------------------------------
        # Stereo -> mono
        # ----------------------------------------------------

        if data.ndim > 1:

            data = np.mean(
                data,
                axis=1,
                dtype=np.float32,
            )

        # ----------------------------------------------------
        # Float32
        # ----------------------------------------------------

        data = np.asarray(
            data,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Resample
        # ----------------------------------------------------

        if (
            int(sample_rate)
            != self.sample_rate
        ):

            data = librosa.resample(
                data,
                orig_sr=int(
                    sample_rate
                ),
                target_sr=self.sample_rate,
            )

            data = np.asarray(
                data,
                dtype=np.float32,
            )

        return np.ascontiguousarray(
            data,
            dtype=np.float32,
        )

    # ========================================================
    # Load audio file
    # ========================================================

    def load_audio(
        self,
        audio_path,
    ):

        audio_path = Path(
            audio_path
        )

        if not audio_path.exists():

            raise FileNotFoundError(
                "Audio file does not exist:\n"
                f"{audio_path}"
            )

        audio, sample_rate = sf.read(
            str(audio_path),
            always_2d=False,
            dtype="float32",
        )

        audio = self.prepare_audio(
            audio,
            int(sample_rate),
        )

        duration = (
            len(audio)
            / self.sample_rate
        )

        return (
            audio,
            duration,
        )

    # ========================================================
    # Transcribe
    # ========================================================

    @torch.inference_mode()
    def transcribe(
        self,
        audio,
        sample_rate=None,
    ) -> dict:

        if not self.loaded:

            self.load()

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        if isinstance(
            audio,
            (str, Path),
        ):

            audio, duration = (
                self.load_audio(
                    audio
                )
            )

        else:

            if sample_rate is None:

                sample_rate = (
                    self.sample_rate
                )

            audio = self.prepare_audio(
                audio,
                int(sample_rate),
            )

            duration = (
                len(audio)
                / self.sample_rate
            )

        if audio.size == 0:

            return {
                "text": "",
                "duration": 0.0,
                "processing_time": 0.0,
                "rtf": 0.0,
                "realtime_factor": 0.0,
                "sample_rate": (
                    self.sample_rate
                ),
                "device": str(
                    self.device
                ),
                "engine": (
                    "shenava_koochik"
                ),
                "decoder": (
                    self.decoder
                ),
                "attention_context": (
                    self.attention_context
                ),
                "peak_vram_gb": None,
            }

        # ----------------------------------------------------
        # CUDA statistics
        # ----------------------------------------------------

        if self.device.type == "cuda":

            torch.cuda.empty_cache()

            torch.cuda.reset_peak_memory_stats(
                self.device
            )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        start_time = perf_counter()

        hypotheses = (
            self.model.transcribe(
                [audio],
                batch_size=1,
                verbose=False,
            )
        )

        processing_time = (
            perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # Decode result
        # ----------------------------------------------------

        result = hypotheses[0]

        if hasattr(
            result,
            "text",
        ):

            text = result.text

        else:

            text = str(
                result
            )

        text = str(
            text or ""
        ).strip()

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        if processing_time > 0:

            rtf = (
                processing_time
                / duration
            )

            realtime_factor = (
                duration
                / processing_time
            )

        else:

            rtf = 0.0
            realtime_factor = 0.0

        # ----------------------------------------------------
        # VRAM
        # ----------------------------------------------------

        peak_vram_gb = None

        if self.device.type == "cuda":

            peak_vram_gb = (
                torch.cuda.max_memory_allocated(
                    self.device
                )
                / (1024 ** 3)
            )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        return {
            "text": text,

            "duration": round(
                duration,
                3,
            ),

            "processing_time": round(
                processing_time,
                3,
            ),

            "rtf": round(
                rtf,
                4,
            ),

            "realtime_factor": round(
                realtime_factor,
                2,
            ),

            "sample_rate": (
                self.sample_rate
            ),

            "device": str(
                self.device
            ),

            "engine": (
                "shenava_koochik"
            ),

            "decoder": (
                self.decoder
            ),

            "attention_context": (
                self.attention_context
            ),

            "peak_vram_gb": (
                round(
                    peak_vram_gb,
                    3,
                )
                if peak_vram_gb is not None
                else None
            ),
        }

    # ========================================================
    # Unload
    # ========================================================

    def unload(
        self,
    ) -> None:

        self.model = None

        self.loaded = False

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        print(
            "Shenava Koochik unloaded."
        )