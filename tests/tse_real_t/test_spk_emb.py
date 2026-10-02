from pathlib import Path
import sys
import json
import numpy as np
import torch
import soundfile as sf

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\Projects\persian_asr")
WESEP_ROOT = PROJECT_ROOT / "models" / "WeSep"

TEST_ROOT = PROJECT_ROOT / "tests" / "tse_real_t"

MODEL_DIR = PROJECT_ROOT / "models" / "REAL-T" / "spk_emb_100"
MIXTURE = TEST_ROOT / "mixture.wav"
ENROLL = TEST_ROOT / "references" / "SPEAKER_00.wav"

RESULTS_DIR = TEST_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = RESULTS_DIR / "SPEAKER_00_overlap_test.wav"

# ---------------------------------------------------------------------
# Overlaps for SPEAKER_00
# ---------------------------------------------------------------------

OVERLAPS = [
    (205.096, 205.163, ["SPEAKER_00", "SPEAKER_01"]),
    (210.293, 210.715, ["SPEAKER_00", "SPEAKER_01"]),
    (212.200, 212.453, ["SPEAKER_00", "SPEAKER_01"]),
    (213.668, 215.018, ["SPEAKER_00", "SPEAKER_01"]),
    (229.852, 231.370, ["SPEAKER_00", "SPEAKER_01"]),
    (231.522, 231.556, ["SPEAKER_00", "SPEAKER_01"]),
    (231.623, 231.708, ["SPEAKER_00", "SPEAKER_01"]),
    (231.725, 231.758, ["SPEAKER_00", "SPEAKER_02"]),
    (237.563, 240.213, ["SPEAKER_00", "SPEAKER_01"]),
    (243.537, 243.655, ["SPEAKER_00", "SPEAKER_01"]),
    (243.689, 243.723, ["SPEAKER_00", "SPEAKER_01"]),
    (243.740, 243.757, ["SPEAKER_00", "SPEAKER_01"]),
    (260.142, 261.121, ["SPEAKER_00", "SPEAKER_02"]),
    (261.475, 261.560, ["SPEAKER_00", "SPEAKER_02"]),
    (264.040, 264.783, ["SPEAKER_00", "SPEAKER_02"]),
    (265.930, 267.652, [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_02"
    ]),
]

CONTEXT = 1.0  # seconds before/after each overlap

# ---------------------------------------------------------------------
# Import WeSep
# ---------------------------------------------------------------------

sys.path.insert(0, str(WESEP_ROOT))

import wesep

# ---------------------------------------------------------------------
# Basic checks
# ---------------------------------------------------------------------

print("=" * 70)
print("REAL-T / SPEAKER_00 OVERLAP TEST")
print("=" * 70)

print(f"Model      : {MODEL_DIR}")
print(f"Mixture    : {MIXTURE}")
print(f"Enrollment : {ENROLL}")
print(f"Output     : {OUTPUT}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print()
print("GPU:", torch.cuda.get_device_name(0))

for path in [MODEL_DIR, MIXTURE, ENROLL]:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

# ---------------------------------------------------------------------
# Load mixture
# ---------------------------------------------------------------------

audio, sr = sf.read(
    str(MIXTURE),
    dtype="float32",
)

if audio.ndim > 1:
    audio = audio.mean(axis=1)

if sr != 16000:
    raise RuntimeError(f"Expected 16 kHz, got {sr} Hz.")

total_duration = len(audio) / sr

print()
print(f"Mixture duration: {total_duration:.3f} sec")
print(f"Context         : +/- {CONTEXT:.1f} sec")

# ---------------------------------------------------------------------
# Merge overlapping / nearby intervals
# ---------------------------------------------------------------------

intervals = []

for start, end, speakers in OVERLAPS:

    if "SPEAKER_00" not in speakers:
        continue

    start = max(0.0, start - CONTEXT)
    end = min(total_duration, end + CONTEXT)

    intervals.append([start, end])

intervals.sort()

merged = []

for start, end in intervals:

    if not merged:
        merged.append([start, end])
        continue

    previous_start, previous_end = merged[-1]

    if start <= previous_end:
        merged[-1][1] = max(previous_end, end)
    else:
        merged.append([start, end])

print()
print(f"Original overlap events : {len(intervals)}")
print(f"Merged test segments    : {len(merged)}")

for i, (start, end) in enumerate(merged, 1):
    print(
        f"  Segment {i:02d}: "
        f"{start:.3f} -> {end:.3f} "
        f"({end - start:.3f}s)"
    )

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------

print()
print("Loading WeSep model...")

model = wesep.load_model_local(str(MODEL_DIR))

model.set_resample_rate(16000)
model.set_vad(False)
model.set_device("cuda")
model.set_output_norm(True)

print("Model ready.")

# ---------------------------------------------------------------------
# TSE each overlap segment
# ---------------------------------------------------------------------

outputs = []

print()
print("Running TSE...")
print()

for i, (start, end) in enumerate(merged, 1):

    start_sample = int(start * sr)
    end_sample = int(end * sr)

    segment = audio[start_sample:end_sample]

    segment_path = TEST_ROOT / f"_overlap_segment_{i:02d}.wav"

    sf.write(
        str(segment_path),
        segment,
        sr,
        subtype="PCM_16",
    )

    print(
        f"[{i:02d}/{len(merged)}] "
        f"{start:.3f}s -> {end:.3f}s "
        f"({len(segment) / sr:.3f}s)"
    )

    try:

        with torch.inference_mode():

            speech = model.extract_speech(
                str(segment_path),
                str(ENROLL),
            )

        if torch.is_tensor(speech):
            speech = speech.detach().cpu().numpy()

        speech = speech.squeeze()

        outputs.append(speech)

    finally:

        if segment_path.exists():
            segment_path.unlink()

        torch.cuda.empty_cache()

# ---------------------------------------------------------------------
# Concatenate extracted segments
# ---------------------------------------------------------------------

speech = np.concatenate(outputs)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

sf.write(
    str(OUTPUT),
    speech,
    sr,
    subtype="PCM_16",
)

print()
print("=" * 70)
print("SUCCESS")
print("=" * 70)

print(f"Output   : {OUTPUT}")
print(f"Duration : {len(speech) / sr:.3f} sec")