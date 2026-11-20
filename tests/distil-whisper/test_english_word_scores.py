
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)


DEFAULT_MODEL = Path("models/distil-whisper-small.en")


def get_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda:0", torch.float16

    return "cpu", torch.float32


def split_words(tokenizer, token_ids, token_scores):
    """
    Reconstruct words from Whisper BPE tokens.

    Each word receives:
        - token_count
        - average_logprob
        - geometric_mean_confidence
        - minimum_token_confidence
        - token-level details

    Confidence is based on the model's generated-token
    probabilities. It is not a calibrated probability
    that the word is correct.
    """

    special_ids = set(tokenizer.all_special_ids)

    tokens = tokenizer.convert_ids_to_tokens(
        token_ids.tolist()
    )

    words = []
    current = None

    for token_id, token, logprob in zip(
        token_ids.tolist(),
        tokens,
        token_scores.tolist(),
    ):
        if token_id in special_ids:
            continue

        # Whisper uses byte-level BPE. "Ġ" usually marks
        # the beginning of a new word.
        starts_word = token.startswith("Ġ")

        piece = token.removeprefix("Ġ")

        # Ignore empty token pieces.
        if not piece:
            continue

        # Punctuation should normally remain attached
        # to the preceding word.
        punctuation_only = not any(
            character.isalnum()
            for character in piece
        )

        if current is None:
            current = {
                "pieces": [],
                "logprobs": [],
                "tokens": [],
            }

        elif starts_word and not punctuation_only:
            words.append(current)

            current = {
                "pieces": [],
                "logprobs": [],
                "tokens": [],
            }

        current["pieces"].append(piece)
        current["logprobs"].append(float(logprob))
        current["tokens"].append(token)

    if current is not None:
        words.append(current)

    results = []

    for item in words:
        word = tokenizer.convert_tokens_to_string(
            item["tokens"]
        ).strip()

        if not word:
            continue

        logprobs = item["logprobs"]

        average_logprob = sum(logprobs) / len(logprobs)

        # exp(mean(log probability)) is the geometric
        # mean of the selected token probabilities.
        geometric_confidence = float(
            torch.exp(
                torch.tensor(
                    average_logprob,
                    dtype=torch.float64,
                )
            ).item()
        )

        token_confidences = [
            float(torch.exp(torch.tensor(lp)).item())
            for lp in logprobs
        ]

        results.append(
            {
                "word": word,
                "token_count": len(logprobs),
                "average_logprob": round(
                    average_logprob, 6
                ),
                "confidence": round(
                    geometric_confidence, 6
                ),
                "minimum_token_confidence": round(
                    min(token_confidences), 6
                ),
                "tokens": item["tokens"],
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Distil-Whisper English ASR with "
            "token and word confidence scores"
        )
    )

    parser.add_argument(
        "audio",
        type=Path,
        help="Input audio file",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Local model directory",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help=(
            "Display threshold from 0 to 1. "
            "This is a model score, not calibrated accuracy."
        ),
    )

    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )

    args = parser.parse_args()

    audio_path = args.audio.resolve()
    model_path = args.model.resolve()

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}"
        )

    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0 and 1")

    device, dtype = get_device()

    print("=" * 78)
    print("DISTIL WHISPER - ENGLISH WORD SCORE TEST")
    print("=" * 78)
    print(f"Audio : {audio_path}")
    print(f"Model : {model_path}")
    print(f"Device: {device}")
    print()

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    print("Loading model...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
    ).to(device)

    model.eval()

    # Load audio through the same feature extractor
    # used by Whisper.
    import soundfile as sf
    import librosa
    import numpy as np

    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=False,
    )

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sample_rate != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )
        sample_rate = 16000

    audio = np.asarray(audio, dtype=np.float32)

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(
        device=device,
        dtype=dtype,
    )

    print()
    print("Generating transcription and token scores...")

    start_time = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            input_features,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )

    elapsed = time.perf_counter() - start_time

    # Compute log probabilities for the tokens selected
    # by the generation process.
    transition_scores = model.compute_transition_scores(
        output.sequences,
        output.scores,
        normalize_logits=True,
    )

    sequence = output.sequences[0]

    # For encoder-decoder generation, the first sequence
    # token is normally the decoder-start token.
    generated_ids = sequence[1:]

    # Keep IDs and scores aligned defensively.
    score_values = transition_scores[0]

    aligned_length = min(
        len(generated_ids),
        len(score_values),
    )

    generated_ids = generated_ids[-aligned_length:]
    score_values = score_values[-aligned_length:]

    tokenizer = processor.tokenizer

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    words = split_words(
        tokenizer,
        generated_ids,
        score_values,
    )

    # English lexical candidates: words containing at
    # least one ASCII letter or digit.
    english_candidates = []

    for item in words:
        lexical = re.sub(
            r"[^A-Za-z0-9]+",
            "",
            item["word"],
        )

        if lexical:
            candidate = dict(item)
            candidate["lexical_form"] = lexical
            english_candidates.append(candidate)

    selected_words = [
        word
        for word in english_candidates
        if word["confidence"] >= args.min_confidence
    ]

    print()
    print("=" * 78)
    print("TRANSCRIPTION")
    print("=" * 78)
    print(text or "[Empty transcription]")

    print()
    print("=" * 78)
    print("ENGLISH WORD CANDIDATES")
    print("=" * 78)

    print(
        f"{'WORD':<24}"
        f"{'CONFIDENCE':>14}"
        f"{'AVG LOGPROB':>16}"
        f"{'TOKENS':>10}"
    )

    print("-" * 78)

    for item in selected_words:
        print(
            f"{item['word']:<24}"
            f"{item['confidence']:>14.4f}"
            f"{item['average_logprob']:>16.4f}"
            f"{item['token_count']:>10}"
        )

    print("-" * 78)
    print(f"Total English candidates: {len(english_candidates)}")
    print(f"Candidates above threshold: {len(selected_words)}")
    print(f"Inference time: {elapsed:.2f} seconds")

    output_path = (
        args.json.resolve()
        if args.json
        else audio_path.with_name(
            audio_path.stem + "_word_scores.json"
        )
    )

    result = {
        "audio": str(audio_path),
        "model": str(model_path),
        "device": device,
        "transcription": text,
        "inference_seconds": round(elapsed, 3),
        "min_confidence_threshold": args.min_confidence,
        "score_definition": (
            "Geometric mean of selected generated-token "
            "probabilities. Not a calibrated probability "
            "of word correctness."
        ),
        "english_candidates": english_candidates,
        "selected_candidates": selected_words,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"JSON saved: {output_path}")


if __name__ == "__main__":
    main()
