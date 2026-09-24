# Persian ASR

Offline Persian Speech-to-Text project.

## Current Engine

Whisper Persian V4

Model:

nezamisafa/whisper-persian-v4

## Architecture

```text
persian_asr/
│
├── config/
│   └── config.yaml
│
├── models/
│   └── whisper_persian_v4/
│
├── engines/
│   └── whisper_persian_v4/
│       ├── __init__.py
│       └── engine.py
│
├── tests/
│   ├── samples/
│   │   └── male1.wav
│   │
│   └── whisper_persian_v4/
│       ├── __init__.py
│       ├── test_model.py
│       ├── test_inference.py
│       └── results/
│
├── requirements/
│   ├── README.txt
│   └── whisper_persian_v4.txt
│
├── .gitignore
├── README.md
└── main.py