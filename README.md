# Persian Meeting ASR

> Offline Persian speech-to-text pipeline for meeting recordings with speaker diarization, optional target speaker extraction, audio-aware ASR grouping, and local AI-generated meeting minutes.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-supported-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Whisper](https://img.shields.io/badge/ASR-Whisper%20Persian%20v4-orange.svg)](https://github.com/openai/whisper)
[![PyAnnote](https://img.shields.io/badge/Diarization-PyAnnote-purple.svg)](https://github.com/pyannote/pyannote-audio)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black.svg)](https://ollama.com/)

## Overview

**Persian Meeting ASR** is an offline-first speech processing pipeline designed for Persian meeting recordings.

The system converts a raw meeting recording into:

1. Speaker-aware audio segments
2. Speaker diarization results
3. Cleaned and validated ASR groups
4. Persian transcription
5. Optional target-speaker extraction
6. Optional overlap speech separation using REAL-T
7. Structured JSON output
8. Human-readable meeting transcript
9. AI-generated meeting minutes using a local LLM

The project is designed around a modular architecture so that individual components can be tested, replaced, or disabled without redesigning the entire pipeline.

---

## Key Features

### Persian ASR

* Persian-focused speech recognition
* Whisper Persian v4 engine
* CUDA acceleration
* Configurable audio preprocessing
* Support for common audio formats
* Chunked transcription for long recordings

### Speaker Diarization

The pipeline identifies speaker activity and assigns speaker labels to different regions of the recording.

Current diarization configuration supports:

* PyAnnote
* Configurable minimum number of speakers
* Configurable maximum number of speakers
* RTTM export
* JSON export
* Speaker-turn export
* Exclusive diarization mode

Example:

```text
SPEAKER_00    00:00:02.10 → 00:00:06.85
SPEAKER_01    00:00:06.92 → 00:00:11.40
SPEAKER_00    00:00:11.51 → 00:00:18.20
```

---

## Two Processing Modes

The system supports two main processing paths.

### Mode 1 — TSE Disabled

```text
Input Audio
    │
    ▼
Preprocessing
    │
    ▼
Speaker Diarization
    │
    ▼
ASR Grouping / Filtering
    │
    ▼
Whisper Persian v4
    │
    ▼
Transcript
    │
    ▼
Optional Meeting Minutes
```

This mode uses the original audio as the transcription source.

It is useful when:

* overlapping speech is limited
* maximum processing speed is preferred
* speaker isolation is not required
* the original recording already has relatively clean speech

### Mode 2 — TSE Enabled

```text
Input Audio
    │
    ▼
Preprocessing
    │
    ▼
Speaker Diarization
    │
    ├───────────────┐
    │               │
    ▼               ▼
Speaker References  Overlap Detection
    │               │
    │               ▼
    │          REAL-T / TSE
    │               │
    └───────┬───────┘
            ▼
    Speaker-Aware Audio
            │
            ▼
    ASR Grouping / Filtering
            │
            ▼
    Whisper Persian v4
            │
            ▼
        Transcript
            │
            ▼
    Optional Meeting Minutes
```

The TSE path is designed for meetings where overlapping speech is important.

It can use:

* speaker reference recordings
* speaker embeddings
* REAL-T
* WeSep / WeSpeaker components
* overlap-region extraction
* timeline-preserving speaker audio

This path requires additional processing and model dependencies.

---

## Why TSE Is Optional

Target Speaker Extraction is computationally more expensive than standard diarization + ASR.

Therefore, the pipeline intentionally allows TSE to be disabled.

### TSE OFF

Advantages:

* faster processing
* simpler pipeline
* fewer model dependencies
* lower GPU requirements
* suitable for normal meetings

Trade-off:

* overlapping speakers may contaminate each other's transcription

### TSE ON

Advantages:

* improved handling of overlapping speech
* speaker-focused audio
* better separation of simultaneous speakers
* useful for difficult meeting recordings

Trade-offs:

* higher processing cost
* additional models
* more GPU memory requirements
* more complex processing pipeline

---

# Pipeline Architecture

```text
                         ┌─────────────────────┐
                         │    Input Recording  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Audio Preprocessing │
                         │ 16 kHz / Mono       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Speaker Diarization│
                         │      PyAnnote       │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
              ┌─────────────┐              ┌─────────────┐
              │   TSE OFF   │              │   TSE ON    │
              └──────┬──────┘              └──────┬──────┘
                     │                             │
                     │                      ┌──────▼──────┐
                     │                      │ Speaker/TSE │
                     │                      │ Processing  │
                     │                      └──────┬──────┘
                     │                             │
                     └──────────────┬──────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  ASR Grouping       │
                         │ Filtering/Validation│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Whisper Persian v4  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Persian Transcript  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Local Ollama / Qwen │
                         │ Meeting Minutes     │
                         └─────────────────────┘
```

---

# Processing Stages

## 1. Audio Preprocessing

Input recordings are normalized before downstream processing.

Current configuration supports:

* resampling to 16 kHz
* mono conversion
* configurable audio extensions
* optional enhancement stage

Supported formats include:

```text
.wav
.mp3
.flac
.m4a
.ogg
```

Current default:

```yaml
sample_rate: 16000
channels: 1
language: fa
task: transcribe
```

---

## 2. Audio Chunking

Long recordings are divided into manageable regions.

The chunking stage supports:

* maximum chunk duration
* minimum chunk duration
* silence-aware splitting
* padding before speech
* padding after speech
* fallback splitting

Example configuration:

```yaml
chunking:
  enabled: true
  max_chunk_seconds: 30
  min_chunk_seconds: 0.5
  silence_top_db: 30
  min_silence_seconds: 0.35
  padding_before_seconds: 0.20
  padding_after_seconds: 0.20
```

---

## 3. Speaker Diarization

PyAnnote is used to estimate speaker activity.

The diarization stage produces speaker turns that are later consumed by the grouping and ASR layers.

Outputs can include:

```text
RTTM
JSON
Speaker Turns
```

Example:

```yaml
diarization:
  enabled: true
  engine: pyannote
  min_speakers: 3
  max_speakers: 5
  use_exclusive: true
```

---

# 4. ASR Grouping and Filtering

One of the important parts of the pipeline is the intermediate grouping layer.

Raw diarization turns are not always suitable for direct transcription.

Real recordings may contain:

* extremely short speaker turns
* accidental speaker switches
* silence-only regions
* low-energy noise
* short overlap regions
* false-positive diarization segments
* fragmented speech from the diarization engine

The grouping layer therefore attempts to produce meaningful ASR units before sending audio to Whisper.

### Grouping capabilities

* merge nearby turns from the same speaker
* limit maximum group duration
* preserve timeline information
* ignore extremely short turns
* bridge short false speaker turns
* filter short overlap regions
* validate actual audio duration
* reject silence-only groups
* reject noise-only groups
* preserve contiguous group numbering

Example configuration:

```yaml
asr:
  grouping:
    enabled: true

    max_gap_seconds: 1.50
    max_chunk_seconds: 30.0
    min_group_seconds: 0.50
    preserve_gaps: true

    short_turn_filter:
      enabled: true
      ignore_below_seconds: 0.50
      bridge_same_speaker: true
      bridge_max_gap_seconds: 1.50

    overlap_filter:
      enabled: true
      ignore_below_seconds: 0.50
      do_not_break_group: true

    noise_filter:
      enabled: true
      max_noise_turn_seconds: 0.50
      bridge_same_speaker: true
      min_rms_ratio: 0.08
      min_rms: 0.0005
      min_active_ratio: 0.02

    audio_validation:
      enabled: true
      validate_actual_duration: true
      min_audio_seconds: 0.20
      min_active_ratio: 0.02
      reject_silence_only: true
      reject_noise_only: true
```

This stage is particularly important because diarization output should not automatically be treated as valid speech.

---

# 5. Whisper Persian v4

The ASR engine uses a Persian-focused Whisper model.

The transcription stage receives validated audio groups and produces Persian text.

Conceptually:

```text
Audio Group
     │
     ▼
Whisper Persian v4
     │
     ▼
Persian Text
     │
     ▼
Speaker + Timestamp Metadata
```

The system is designed so that the ASR engine is isolated from the rest of the pipeline.

This makes it possible to replace or benchmark the ASR engine independently.

---

# 6. Speaker Isolation

Speaker isolation is available as an additional processing layer.

When enabled, the system can generate speaker-specific audio while preserving the original timeline.

Configuration:

```yaml
speaker_isolation:
  enabled: false
  preserve_timeline: true
  references_dir: output/references
  auto_create_references: true
  overwrite_references: true
  reference_max_seconds: 8.0
  reference_min_seconds: 1.0
  save_timeline: true
  output_dir: speakers
```

Speaker references can be generated automatically from diarization results.

---

# 7. REAL-T / Target Speaker Extraction

The optional TSE engine is designed to process difficult overlap regions.

Configuration:

```yaml
overlap_separation:
  enabled: false
  engine: real_t
  model_path: models/REAL-T/spk_emb_100
  wesep_path: models/WeSep
  wespeaker_path: models/WeSpeaker
  sample_rate: 16000
  device: cuda
  dtype: float16
```

The processing pipeline can identify overlap regions and apply target speaker extraction only where required.

This avoids unnecessarily processing the entire recording through the expensive TSE stage.

---

# 8. Debug Audio

The pipeline provides extensive debugging support.

Debug outputs can include:

```text
Speaker references
TSE inputs
TSE outputs
Overlap regions
ASR chunks
```

Example:

```yaml
debug:
  audio:
    enabled: true
    root_dir: output/debug_audio
    save_speaker_references: true
    save_tse_inputs: true
    save_tse_outputs: true
    save_tse_overlap: true
    save_asr_chunks: true
```

Debug data is intended for development and troubleshooting and should normally not be committed to the repository.

---

# 9. Local Meeting Minutes with Ollama

After transcription, the project can optionally generate structured meeting minutes using a local LLM.

Current configuration:

```yaml
ollama:
  model: qwen2.5:14b

meeting_minutes:
  enabled: true

  ollama:
    model: qwen2.5:14b
    host: http://127.0.0.1:11434
    temperature: 0.1
```

The architecture is:

```text
Persian Transcript
       │
       ▼
Prompt Template
       │
       ▼
Local Ollama
       │
       ▼
Qwen 2.5 14B
       │
       ▼
Meeting Minutes
```

Because the model runs locally, the meeting transcript can remain on the local machine.

This makes the stage suitable for workflows where sending meeting recordings or transcripts to external cloud APIs is undesirable.

---

# Meeting Minutes Prompt

The meeting-minutes prompt is configurable through YAML.

The prompt should explicitly tell the LLM that the transcript is generated by ASR and may contain recognition or hearing errors.

For example:

```text
The transcript was generated automatically by an ASR system.
It may contain speech-recognition errors, missing words, incorrect words,
speaker attribution errors, punctuation errors, or other transcription artifacts.

Use the transcript as the primary source, but interpret it carefully.
Do not invent information that is not supported by the transcript.
```

For Persian meetings, the prompt should also explicitly require Persian output.

---

# Project Structure

Current project structure:

```text
persian-meeting-asr/
│
├── config/
│   ├── config.yaml
│   ├── ui_settings.json
│   └── __init__.py
│
├── engines/
│   ├── diarization/
│   │   ├── engine.py
│   │   └── ...
│   │
│   ├── overlap_separation/
│   │   ├── engine.py
│   │   └── ...
│   │
│   ├── speaker_isolation/
│   │   ├── engine.py
│   │   └── ...
│   │
│   ├── whisper_persian_v4/
│   │   ├── engine.py
│   │   ├── __init__.py
│   │   └── ...
│   │
│   ├── asr_grouping.py
│   ├── audio_chunker.py
│   └── __init__.py
│
├── pipeline/
│   └── meeting_pipeline.py
│
├── tests/
│   ├── diarization/
│   ├── meeting_minutes/
│   ├── samples/
│   ├── speaker_isolation/
│   ├── tse_real_t/
│   └── whisper_persian_v4/
│
├── Tkinter_ui/
│
├── ui/
│   ├── components/
│   ├── pages/
│   └── ...
│
├── requirements/
│   ├── current-env.txt
│   ├── main_packages.txt
│   ├── pyannote_requirements.txt
│   ├── remaining_requirements.txt
│   ├── tse_real_t_packages.txt
│   ├── whisper_requirements.txt
│   └── README.txt
│
├── input/
├── output/
│
├── main.py
├── README.md
└── .gitignore
```

Large model files, virtual environments, generated debug audio, recordings, and runtime outputs should remain outside normal source control.

---

# Installation

## Requirements

Recommended environment:

* Windows or Linux
* Python 3.10+
* NVIDIA GPU with CUDA support
* Git
* FFmpeg
* PyTorch
* PyAnnote
* Whisper-compatible dependencies
* Optional: Ollama
* Optional: REAL-T / WeSep / WeSpeaker dependencies

---

## Clone

```bash
git clone https://github.com/<your-username>/persian-meeting-asr.git
cd persian-meeting-asr
```

---

## Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install Dependencies

The project separates dependency lists according to the processing components.

For example:

```powershell
pip install -r requirements/main_packages.txt
```

Additional components can be installed from:

```text
requirements/pyannote_requirements.txt
requirements/whisper_requirements.txt
requirements/tse_real_t_packages.txt
requirements/remaining_requirements.txt
```

The exact installation order may depend on the CUDA/PyTorch environment.

---

# Configuration

The main configuration file is:

```text
config/config.yaml
```

The configuration controls:

* ASR
* diarization
* audio preprocessing
* chunking
* TSE
* speaker isolation
* grouping
* debug output
* Ollama
* meeting minutes
* device and precision
* output formats

---

# Basic Configuration

A minimal ASR configuration looks like:

```yaml
engine:
  name: whisper_persian_v4

model:
  path: models/whisper_persian_v4

audio:
  sample_rate: 16000
  channels: 1
  language: fa
  task: transcribe

device:
  type: cuda
  dtype: float16
```

---

# Running the Pipeline

Place a recording inside:

```text
input/
```

For example:

```text
input/
└── meeting.wav
```

Then run:

```powershell
python main.py
```

The exact CLI/UI workflow may evolve as the project develops.

---

# Output

A processed meeting can produce files such as:

```text
output/
└── meeting/
    ├── meeting_meeting.txt
    ├── meeting_meeting.json
    └── ...
```

The JSON output is intended for machine-readable processing.

The TXT output is intended for human-readable transcription.

Additional RTTM, speaker-turn, isolation, TSE, and debugging files can be generated depending on configuration.

---

# Testing

The repository contains component-level tests for the major processing stages.

## Diarization

```text
tests/diarization/
```

Includes:

* diarization test scripts
* sample audio
* RTTM output
* JSON output
* temporary workspace

## Whisper Persian v4

```text
tests/whisper_persian_v4/
```

Contains inference and model tests.

## Speaker Isolation

```text
tests/speaker_isolation/
```

Used for testing speaker-specific audio generation.

## REAL-T

```text
tests/tse_real_t/
```

Contains TSE-related experiments and tests.

## Meeting Minutes

```text
tests/meeting_minutes/
```

Contains the local Ollama/Qwen meeting-minutes test.

Example:

```powershell
python tests/meeting_minutes/test_meeting_minutes_qwen.py
```

---

# Ollama Setup

The meeting-minutes stage requires a running local Ollama server.

Start Ollama:

```powershell
ollama serve
```

Verify installed models:

```powershell
ollama list
```

If the Qwen model is not available:

```powershell
ollama pull qwen2.5:14b
```

The test communicates with:

```text
http://127.0.0.1:11434
```

The interactive command:

```powershell
ollama run qwen2.5:14b
```

is useful for manually testing the model, but the Python API test primarily requires the Ollama server to be running.

---

# Data Flow Example

For a meeting recording:

```text
meeting.wav
    │
    ▼
16 kHz Mono Audio
    │
    ▼
PyAnnote Diarization
    │
    ▼
Speaker Turns
    │
    ▼
Short-turn / Noise / Overlap Filtering
    │
    ▼
ASR Groups
    │
    ▼
Whisper Persian v4
    │
    ▼
Persian Transcript
    │
    ▼
Qwen 2.5 14B
    │
    ▼
Meeting Minutes
```

---

# Error Handling and Quality Control

The pipeline includes several mechanisms designed to prevent invalid audio regions from reaching the ASR stage.

### Short-turn filtering

Very short diarization turns can be ignored when they are unlikely to represent meaningful speech.

### Noise filtering

Low-energy regions can be rejected based on RMS and active-audio measurements.

### Silence validation

Groups containing only silence can be rejected.

### Actual-duration validation

The pipeline can validate the actual extracted waveform instead of relying only on diarization timestamps.

### Group bridging

A short invalid turn can be ignored while preserving continuity between surrounding turns from the same speaker.

### Contiguous group numbering

Rejected groups should not create gaps in the final debug/group numbering.

---

# Debugging

For development, enable:

```yaml
debug:
  audio:
    enabled: true
    save_speaker_references: true
    save_tse_inputs: true
    save_tse_outputs: true
    save_tse_overlap: true
    save_asr_chunks: true
```

This allows individual processing stages to be inspected independently.

For production use, debug audio generation can be disabled.

---

# Privacy

This project is designed with an offline-first architecture.

The main processing stages can run locally:

```text
Audio
  ↓
Local Diarization
  ↓
Local ASR
  ↓
Local TSE
  ↓
Local Transcript
  ↓
Local Ollama
```

No cloud transcription API is required by the core architecture.

However, individual model downloads or external package repositories may still require network access during installation.

---

# Security and Repository Hygiene

Do not commit:

```text
.venv/
models/
input/
output/
output/debug_audio/
output/.tse_tmp/
__pycache__/
*.log
.env
API keys
access tokens
private recordings
```

Before pushing the repository:

```powershell
git status
```

Review all staged files carefully.

GitHub recommends using repository security features such as secret scanning, push protection, and dependency/security alerts where appropriate.

---

# Current Capabilities

| Component                    | Status                     |
| ---------------------------- | -------------------------- |
| Persian ASR                  | Implemented                |
| Whisper Persian v4           | Implemented                |
| PyAnnote diarization         | Implemented                |
| RTTM export                  | Implemented                |
| JSON output                  | Implemented                |
| Speaker turns                | Implemented                |
| Audio preprocessing          | Implemented                |
| Audio chunking               | Implemented                |
| ASR grouping                 | Implemented                |
| Short-turn filtering         | Implemented                |
| Noise filtering              | Implemented                |
| Audio validation             | Implemented                |
| Speaker isolation            | Implemented                |
| REAL-T integration           | Implemented / Experimental |
| Overlap processing           | Implemented / Experimental |
| Ollama integration           | Implemented                |
| Qwen 2.5 14B meeting minutes | Implemented / Experimental |
| Desktop UI                   | Implemented                |
| Offline workflow             | Supported                  |

---

# Development Roadmap

Potential future improvements include:

* improved Persian ASR benchmarking
* automatic speaker-name assignment
* improved overlap detection
* more robust TSE selection
* better diarization/ASR alignment
* automatic punctuation restoration
* Persian text normalization
* confidence scoring
* transcript quality metrics
* automatic action-item extraction
* decision and task tracking
* meeting topic segmentation
* improved local LLM prompting
* model benchmarking across Persian ASR engines
* automated regression tests
* packaging and deployment improvements

---

# Design Principles

The project follows several architectural principles:

### Modular

Each major processing stage is isolated behind its own engine.

### Configurable

Runtime behavior is controlled through YAML rather than hard-coded values.

### Offline-first

Core processing is designed to run locally.

### Debuggable

Intermediate audio and metadata can be saved for inspection.

### Testable

Major processing components have dedicated tests.

### Replaceable

ASR, diarization, TSE, and LLM components can evolve independently.

---

# Project Status

This project is under active development.

The architecture is currently focused on reliable Persian meeting transcription, speaker-aware processing, overlap handling, and local meeting-minutes generation.

Some components, particularly TSE/REAL-T and local LLM meeting-minutes generation, should be considered experimental until they have been benchmarked across a larger collection of real-world Persian meetings.

---

# Contributing

Contributions, bug reports, experiments, and benchmarking results are welcome.


---

# License

```text
Not desided yet
```
---

# Acknowledgements

This project builds on open-source technologies and research from the speech and machine-learning community, including:

* Whisper
* PyAnnote Audio
* PyTorch
* REAL-T / target speaker extraction research
* WeSep
* WeSpeaker
* Ollama
* Qwen

Please review and comply with the individual licenses and terms of the models, libraries, and external projects used by your deployment.

---

# Citation

If this project is eventually used in research or a published technical system, a formal citation entry can be added here.

```bibtex
@software{persian_meeting_asr,
  title  = {Persian Meeting ASR},
  author = {َAmin},
  year   = {2026},
  url    = {https://github.com/amin13m/persian-meeting-asr.git}
}
```

---

## Repository

**Recommended repository name:**

```text
persian-meeting-asr
```

The repository is intended to provide a complete, modular pipeline for **Persian meeting transcription with speaker awareness, optional target speaker extraction, overlap processing, and local AI-assisted meeting summarization.**
