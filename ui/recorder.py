
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    """
    Robust Windows microphone recorder.

    Features:
        - Automatic input-device detection
        - WASAPI preferred on Windows
        - Fallback to MME / DirectSound
        - Uses device-native sample rate when required
        - Resamples final audio to 16 kHz
        - Mono output
        - PCM 16-bit WAV
        - Thread-safe frame collection
    """

    TARGET_SAMPLE_RATE = 16000

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
    ):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.dtype = dtype

        self._recording = False
        self._stream = None
        self._thread = None
        self._frames: list[np.ndarray] = []

        self._lock = threading.Lock()

        # Actual rate used by the microphone stream.
        self._recording_sample_rate = self.sample_rate

        # Selected input device.
        self._device = None

        # Selected host API name.
        self._host_api = None

    # ============================================================
    # Properties
    # ============================================================

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def device(self):
        return self._device

    @property
    def host_api(self):
        return self._host_api

    @property
    def recording_sample_rate(self) -> int:
        return self._recording_sample_rate

    # ============================================================
    # Device helpers
    # ============================================================

    @staticmethod
    def list_input_devices() -> list[dict]:
        """
        Return available input devices.

        Each item contains:
            index
            name
            channels
            default_samplerate
            hostapi
        """

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        result = []

        for index, device in enumerate(devices):
            max_input_channels = int(
                device.get("max_input_channels", 0)
            )

            if max_input_channels <= 0:
                continue

            hostapi_index = device.get("hostapi")

            hostapi_name = "Unknown"

            if hostapi_index is not None:
                try:
                    hostapi_name = hostapis[hostapi_index]["name"]
                except Exception:
                    pass

            result.append(
                {
                    "index": index,
                    "name": device.get("name", f"Device {index}"),
                    "channels": max_input_channels,
                    "default_samplerate": float(
                        device.get("default_samplerate", 16000)
                    ),
                    "hostapi": hostapi_name,
                }
            )

        return result

    @staticmethod
    def _get_host_api_name(device_index: int) -> str:
        """Return host API name for a device."""

        try:
            device_info = sd.query_devices(device_index)
            hostapi_index = device_info["hostapi"]
            return sd.query_hostapis(hostapi_index)["name"]
        except Exception:
            return "Unknown"

    @classmethod
    def _find_preferred_input_device(cls):
        """
        Find a suitable microphone.

        Preference order on Windows:
            1. WASAPI microphone
            2. DirectSound microphone
            3. MME microphone
            4. Any available input device
        """

        devices = cls.list_input_devices()

        if not devices:
            raise RuntimeError(
                "No input microphone device was found."
            )

        # --------------------------------------------------------
        # Preferred host APIs
        # --------------------------------------------------------

        preferred_hostapis = [
            "Windows WASAPI",
            "Windows DirectSound",
            "MME",
        ]

        # --------------------------------------------------------
        # First pass:
        # Prefer microphones whose name contains Microphone
        # or Microphone Array.
        # --------------------------------------------------------

        microphone_keywords = (
            "microphone",
            "mic",
        )

        for preferred_api in preferred_hostapis:
            for device in devices:
                if device["hostapi"] != preferred_api:
                    continue

                name = device["name"].lower()

                if any(
                    keyword in name
                    for keyword in microphone_keywords
                ):
                    return device

        # --------------------------------------------------------
        # Second pass:
        # Any input device from preferred host APIs.
        # --------------------------------------------------------

        for preferred_api in preferred_hostapis:
            for device in devices:
                if device["hostapi"] == preferred_api:
                    return device

        # --------------------------------------------------------
        # Last fallback.
        # --------------------------------------------------------

        return devices[0]

    # ============================================================
    # Sample-rate helpers
    # ============================================================

    @staticmethod
    def _supported_sample_rates(
        device_index: int,
        channels: int,
        dtype: str,
    ) -> list[int]:
        """
        Test common microphone sample rates.

        Returns rates that PortAudio accepts for the device.
        """

        candidates = [
            16000,
            32000,
            44100,
            48000,
            22050,
            24000,
        ]

        supported = []

        for rate in candidates:
            try:
                sd.check_input_settings(
                    device=device_index,
                    channels=channels,
                    dtype=dtype,
                    samplerate=rate,
                )

                supported.append(rate)

            except Exception:
                continue

        return supported

    @classmethod
    def _select_sample_rate(
        cls,
        device_index: int,
        channels: int,
        dtype: str,
    ) -> int:
        """
        Select the best recording sample rate.

        Preference:
            1. 16 kHz
            2. 48 kHz
            3. 44.1 kHz
            4. Device native rate
            5. Any rate accepted by PortAudio
        """

        supported = cls._supported_sample_rates(
            device_index=device_index,
            channels=channels,
            dtype=dtype,
        )

        if not supported:
            raise RuntimeError(
                "No compatible sample rate was found "
                "for the selected microphone."
            )

        # Whisper target rate is preferred.
        if 16000 in supported:
            return 16000

        # Common native rates.
        for rate in (48000, 44100, 32000, 24000, 22050):
            if rate in supported:
                return rate

        return supported[0]

    # ============================================================
    # Start
    # ============================================================

    def start(self) -> None:
        """
        Start microphone recording.

        The method tries the preferred microphone and host APIs.
        """

        if self._recording:
            return

        # Make sure a previous stream is completely closed.
        self._cleanup_stream()

        with self._lock:
            self._frames = []

        devices = self.list_input_devices()

        if not devices:
            raise RuntimeError(
                "No microphone/input device is available."
            )

        # --------------------------------------------------------
        # Build candidate device list.
        # --------------------------------------------------------

        candidates = []

        preferred_hostapis = [
            "Windows WASAPI",
            "Windows DirectSound",
            "MME",
        ]

        microphone_keywords = (
            "microphone",
            "mic",
        )

        # First: preferred API + microphone name.
        for api in preferred_hostapis:
            for device in devices:
                if device["hostapi"] != api:
                    continue

                name = device["name"].lower()

                if any(
                    keyword in name
                    for keyword in microphone_keywords
                ):
                    candidates.append(device)

        # Second: preferred API + any input.
        for api in preferred_hostapis:
            for device in devices:
                if device["hostapi"] != api:
                    continue

                if device not in candidates:
                    candidates.append(device)

        # Final fallback.
        for device in devices:
            if device not in candidates:
                candidates.append(device)

        errors = []

        # --------------------------------------------------------
        # Try candidate devices.
        # --------------------------------------------------------

        for device in candidates:

            device_index = device["index"]
            device_name = device["name"]
            host_api = device["hostapi"]

            try:
                recording_rate = self._select_sample_rate(
                    device_index=device_index,
                    channels=self.channels,
                    dtype=self.dtype,
                )

                # ------------------------------------------------
                # Validate before opening the actual stream.
                # ------------------------------------------------

                sd.check_input_settings(
                    device=device_index,
                    channels=self.channels,
                    dtype=self.dtype,
                    samplerate=recording_rate,
                )

                # ------------------------------------------------
                # WASAPI-specific settings.
                # ------------------------------------------------

                extra_settings = None

                if host_api == "Windows WASAPI":
                    try:
                        extra_settings = sd.WasapiSettings(
                            auto_convert=True
                        )
                    except Exception:
                        extra_settings = None

                # ------------------------------------------------
                # Open stream.
                # ------------------------------------------------

                if extra_settings is not None:
                    stream = sd.InputStream(
                        device=device_index,
                        samplerate=recording_rate,
                        channels=self.channels,
                        dtype=self.dtype,
                        callback=self._audio_callback,
                        extra_settings=extra_settings,
                    )
                else:
                    stream = sd.InputStream(
                        device=device_index,
                        samplerate=recording_rate,
                        channels=self.channels,
                        dtype=self.dtype,
                        callback=self._audio_callback,
                    )

                # Start only after successful construction.
                stream.start()

                self._stream = stream
                self._device = device_index
                self._host_api = host_api
                self._recording_sample_rate = recording_rate
                self._recording = True

                print(
                    "Audio recorder started:"
                    f" device={device_index}"
                    f" | {device_name}"
                    f" | host={host_api}"
                    f" | rate={recording_rate}"
                )

                return

            except Exception as exc:
                errors.append(
                    f"[{device_index}] "
                    f"{device_name} "
                    f"({host_api}) -> {exc}"
                )

                # Make absolutely sure failed streams don't remain open.
                self._cleanup_stream()

        # --------------------------------------------------------
        # Nothing worked.
        # --------------------------------------------------------

        self._recording = False
        self._device = None
        self._host_api = None

        error_text = "\n".join(errors)

        raise RuntimeError(
            "Unable to open any microphone input device.\n\n"
            f"{error_text}"
        )

    # ============================================================
    # Audio callback
    # ============================================================

    def _audio_callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ):
        """
        PortAudio callback.

        Never perform expensive processing here.
        """

        if status:
            print(f"Audio recorder: {status}")

        if not self._recording:
            return

        try:
            frame_copy = indata.copy()

            with self._lock:
                self._frames.append(frame_copy)

        except Exception as exc:
            print(
                f"Audio recorder callback error: {exc}"
            )

    # ============================================================
    # Stop
    # ============================================================

    def stop(
        self,
        output_path: str | Path,
    ) -> Path | None:
        """
        Stop recording and save WAV.

        Output:
            Mono
            16 kHz
            PCM 16-bit
        """

        if not self._recording and self._stream is None:
            return None

        # Stop accepting new frames.
        self._recording = False

        self._cleanup_stream()

        # --------------------------------------------------------
        # Copy frames safely.
        # --------------------------------------------------------

        with self._lock:
            frames = list(self._frames)
            self._frames = []

        if not frames:
            self._reset_state()
            return None

        try:
            audio = np.concatenate(
                frames,
                axis=0,
            )

        except Exception as exc:
            self._reset_state()

            raise RuntimeError(
                f"Failed to combine recorded audio: {exc}"
            ) from exc

        # --------------------------------------------------------
        # Convert to mono.
        # --------------------------------------------------------

        if audio.ndim == 2 and audio.shape[1] > 1:
            audio = np.mean(
                audio,
                axis=1,
            )

        elif audio.ndim == 2:
            audio = audio[:, 0]

        # --------------------------------------------------------
        # Ensure float32.
        # --------------------------------------------------------

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # Resample to Whisper's 16 kHz if required.
        # --------------------------------------------------------

        source_rate = self._recording_sample_rate

        if source_rate != self.sample_rate:
            audio = self._resample(
                audio,
                source_rate,
                self.sample_rate,
            )

        # --------------------------------------------------------
        # Clip before PCM conversion.
        # --------------------------------------------------------

        audio = np.clip(
            audio,
            -1.0,
            1.0,
        )

        # --------------------------------------------------------
        # Save WAV.
        # --------------------------------------------------------

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sf.write(
            output_path,
            audio,
            self.sample_rate,
            subtype="PCM_16",
        )

        print(
            "Audio recorder stopped:"
            f" {output_path}"
        )

        self._reset_state()

        return output_path

    # ============================================================
    # Cancel
    # ============================================================

    def cancel(self) -> None:
        """Stop recording and discard captured audio."""

        self._recording = False

        self._cleanup_stream()

        with self._lock:
            self._frames = []

        self._reset_state()

        print("Audio recorder cancelled.")

    # ============================================================
    # Cleanup
    # ============================================================

    def _cleanup_stream(self) -> None:
        """Safely stop and close the active stream."""

        stream = self._stream

        self._stream = None

        if stream is None:
            return

        try:
            stream.stop()
        except Exception:
            pass

        try:
            stream.close()
        except Exception:
            pass

    def _reset_state(self) -> None:
        """Reset recorder state after stop/cancel."""

        self._recording = False
        self._device = None
        self._host_api = None
        self._recording_sample_rate = self.sample_rate

    # ============================================================
    # Resampling
    # ============================================================

    @staticmethod
    def _resample(
        audio: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        """
        Resample audio without requiring another package.

        Uses linear interpolation.

        This is only used when the microphone cannot record
        directly at the target sample rate.
        """

        if source_rate == target_rate:
            return audio

        if audio.size == 0:
            return audio

        source_rate = int(source_rate)
        target_rate = int(target_rate)

        duration = len(audio) / source_rate

        target_length = max(
            1,
            int(round(duration * target_rate)),
        )

        source_positions = np.linspace(
            0.0,
            duration,
            num=len(audio),
            endpoint=False,
        )

        target_positions = np.linspace(
            0.0,
            duration,
            num=target_length,
            endpoint=False,
        )

        resampled = np.interp(
            target_positions,
            source_positions,
            audio,
        )

        return resampled.astype(
            np.float32,
            copy=False,
        )

    # ============================================================
    # Filename
    # ============================================================

    @staticmethod
    def generate_output_path(
        output_dir: str | Path,
    ) -> Path:
        """
        Generate a unique recording filename.
        """

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        return (
            output_dir
            / f"recording_{timestamp}.wav"
        )

