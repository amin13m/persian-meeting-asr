from pathlib import Path
from time import perf_counter

import librosa
import soundfile as sf
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)


class WhisperPersianV4Engine:
    """
    Independent inference engine for:

        nezamisafa/whisper-persian-v4

    Public API:

        engine.load()

        engine.transcribe(audio)

        engine.unload()

    The engine is intentionally unaware of:
        - input directories
        - output directories
        - bulk processing
        - silence detection
        - chunk management

    It receives one audio signal and returns one transcription.
    """

    def __init__(self, config: dict):

        self.config = config

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model_path = Path(
            config["model"]["path"]
        )

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        self.sample_rate = int(
            config["audio"].get(
                "sample_rate",
                16000,
            )
        )

        self.language = config["audio"].get(
            "language",
            "fa",
        )

        self.task = config["audio"].get(
            "task",
            "transcribe",
        )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        device_type = config["device"].get(
            "type",
            "auto",
        ).lower()

        dtype_name = config["device"].get(
            "dtype",
            "float16",
        ).lower()

        self.device = self._resolve_device(
            device_type
        )

        self.dtype = self._resolve_dtype(
            dtype_name
        )

        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        self.processor = None
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

            return torch.device("cuda")

        if device_type == "cpu":

            return torch.device("cpu")

        if device_type == "auto":

            if torch.cuda.is_available():
                return torch.device("cuda")

            return torch.device("cpu")

        raise ValueError(
            f"Unsupported device type: {device_type}"
        )

    # ========================================================
    # DType
    # ========================================================

    def _resolve_dtype(
        self,
        dtype_name: str,
    ) -> torch.dtype:

        if self.device.type == "cpu":

            return torch.float32

        if dtype_name == "float16":

            return torch.float16

        if dtype_name == "float32":

            return torch.float32

        if dtype_name == "bfloat16":

            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "bfloat16 is not supported "
                    "by this CUDA device."
                )

            return torch.bfloat16

        raise ValueError(
            f"Unsupported dtype: {dtype_name}"
        )

    # ========================================================
    # Load
    # ========================================================

    def load(self):

        if self.loaded:
            return

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model directory does not exist:\n"
                f"{self.model_path}"
            )

        print(
            "=" * 60
        )

        print(
            "Loading Whisper Persian V4"
        )

        print(
            f"Model : {self.model_path}"
        )

        print(
            f"Device: {self.device}"
        )

        print(
            f"DType : {self.dtype}"
        )

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        self.processor = (
            AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model = (
            AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.loaded = True

        print(
            "Model loaded successfully."
        )

        print()

    # ========================================================
    # Prepare Audio
    # ========================================================

    def prepare_audio(
        self,
        audio,
        sample_rate: int,
    ):
        """
        Convert arbitrary audio input into:

            mono float32
            target sample rate
        """

        if hasattr(audio, "ndim"):

            data = audio

        else:

            data = audio

        # ----------------------------------------------------
        # Stereo -> mono
        # ----------------------------------------------------

        if data.ndim > 1:

            data = data.mean(
                axis=1
            )

        # ----------------------------------------------------
        # Float32
        # ----------------------------------------------------

        data = data.astype(
            "float32"
        )

        # ----------------------------------------------------
        # Resample
        # ----------------------------------------------------

        if sample_rate != self.sample_rate:

            data = librosa.resample(
                data,
                orig_sr=sample_rate,
                target_sr=self.sample_rate,
            )

        return data

    # ========================================================
    # Load Audio File
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
                f"Audio file does not exist:\n"
                f"{audio_path}"
            )

        audio, sample_rate = sf.read(
            str(audio_path),
            always_2d=False,
        )

        audio = self.prepare_audio(
            audio,
            sample_rate,
        )

        duration = (
            len(audio)
            / self.sample_rate
        )

        return audio, duration

    # ========================================================
    # Transcribe
    # ========================================================

    @torch.inference_mode()
    def transcribe(
        self,
        audio,
        sample_rate=None,
    ):

        if not self.loaded:

            self.load()

        # ----------------------------------------------------
        # Audio input
        # ----------------------------------------------------

        if isinstance(
            audio,
            (str, Path),
        ):

            audio, duration = (
                self.load_audio(audio)
            )

        else:

            if sample_rate is None:

                sample_rate = (
                    self.sample_rate
                )

            audio = self.prepare_audio(
                audio,
                sample_rate,
            )

            duration = (
                len(audio)
                / self.sample_rate
            )

        # ----------------------------------------------------
        # CUDA statistics
        # ----------------------------------------------------

        if self.device.type == "cuda":

            torch.cuda.empty_cache()

            torch.cuda.reset_peak_memory_stats(
                self.device
            )

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )

        input_features = (
            inputs.input_features.to(
                self.device,
                dtype=self.dtype,
            )
        )

        attention_mask = inputs.get(
            "attention_mask"
        )

        if attention_mask is not None:

            attention_mask = (
                attention_mask.to(
                    self.device
                )
            )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        start_time = perf_counter()

        generation_kwargs = {
            "language": self.language,
            "task": self.task,
            "condition_on_prev_tokens": False
        }

        if attention_mask is not None:

            generation_kwargs[
                "attention_mask"
            ] = attention_mask

        generated_ids = (
            self.model.generate(
                input_features,
                **generation_kwargs,
            )
        )

        processing_time = (
            perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        text = (
            self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            .strip()
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        if duration > 0:

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
                torch.cuda
                .max_memory_allocated(
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

            "dtype": str(
                self.dtype
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

    def unload(self):

        self.model = None

        self.processor = None

        self.loaded = False

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        print(
            "Whisper Persian V4 unloaded."
        )