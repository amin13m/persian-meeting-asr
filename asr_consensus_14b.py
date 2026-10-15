from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:14b"

DEFAULT_BATCH_SIZE = 10

TEMPERATURE = 0.0
TOP_P = 0.9

REQUEST_TIMEOUT = 900


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
تو یک متخصص بازسازی و تصحیح متن فارسی حاصل از ASR هستی.

برای هر بخش از یک مکالمه فارسی، دو خروجی مستقل ASR در اختیار داری:

- Whisper Persian
- Shenava Koochik

هدف، تولید دقیق‌ترین نسخه نهایی متن گفتار است.

قوانین بسیار مهم:

1. معنی واقعی صحبت را حفظ کن.
2. Whisper و Shenava را هر دو به عنوان evidence در نظر بگیر.
3. اگر یکی از دو ASR واضحاً اشتباه است، از خروجی دیگر استفاده کن.
4. اگر هر دو خروجی یک مفهوم مشترک را نشان می‌دهند، آن مفهوم را حفظ کن.
5. خطاهای واضح شنیداری را اصلاح کن.
6. کلمات اشتباه شنیده‌شده را با توجه به context اصلاح کن.
7. تکرارهای غیرطبیعی ناشی از ASR را حذف کن.
8. اگر Shenava خروجی‌های تکراری و غیرطبیعی دارد، آن قسمت را وارد متن نهایی نکن.
9. چیزی که هیچ شاهد معقولی در متن ندارد، اختراع نکن.
10. نام اشخاص، شرکت‌ها، محصولات و اصطلاحات تخصصی را با دقت حفظ کن.
11. اعداد و واحدها را تا حد امکان حفظ کن.
12. فارسی نهایی باید طبیعی، خوانا و صحیح باشد.
13. می‌توانی علائم نگارشی مناسب اضافه کنی.
14. می‌توانی اشتباهات واضح دستوری ناشی از ASR را اصلاح کنی.
15. محتوای صحبت را خلاصه نکن.
16. محتوای صحبت را گسترش نده.
17. جمله‌ای را که گوینده نگفته اختراع نکن.
18. speaker را تغییر نده.
19. timestamp را تغییر نده.
20. ترتیب GROUPها را دقیقاً حفظ کن.
21. برای هر GROUP دقیقاً یک خروجی بده.
22. هیچ توضیحی درباره reasoning نده.
23. هیچ markdown نده.
24. فقط خروجی نهایی GROUPها را تولید کن.

فرمت دقیق:

GROUP 0: متن نهایی
GROUP 1: متن نهایی
GROUP 2: متن نهایی

و الی آخر.
"""


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(
    path: Path,
    data: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# READ REAL PIPELINE STRUCTURE
# ============================================================

def load_turns(
    data: dict,
) -> list[dict]:

    transcription = data.get(
        "transcription"
    )

    if not isinstance(
        transcription,
        dict,
    ):
        raise RuntimeError(
            "کلید transcription در JSON پیدا نشد."
        )

    turns = transcription.get(
        "turns"
    )

    if not isinstance(
        turns,
        list,
    ):
        raise RuntimeError(
            "transcription.turns در JSON پیدا نشد."
        )

    return turns


# ============================================================
# ASR EXTRACTION
# ============================================================

def get_whisper_text(
    turn: dict,
) -> str:

    asr = turn.get(
        "asr",
        {},
    )

    if not isinstance(
        asr,
        dict,
    ):
        return ""

    whisper = asr.get(
        "whisper",
        {},
    )

    if isinstance(
        whisper,
        dict,
    ):
        return clean_text(
            whisper.get(
                "text",
                "",
            )
        )

    return clean_text(
        whisper
    )


def get_shenava_text(
    turn: dict,
) -> str:

    asr = turn.get(
        "asr",
        {},
    )

    if not isinstance(
        asr,
        dict,
    ):
        return ""

    shenava = asr.get(
        "shenava",
        {},
    )

    if isinstance(
        shenava,
        dict,
    ):
        return clean_text(
            shenava.get(
                "text",
                "",
            )
        )

    return clean_text(
        shenava
    )


# ============================================================
# NORMALIZED GROUP
# ============================================================

def normalize_turn(
    turn: dict,
) -> dict:

    return {
        "group_index": turn.get(
            "group_index"
        ),
        "speaker": clean_text(
            turn.get(
                "speaker",
                "",
            )
        ),
        "start": turn.get(
            "start"
        ),
        "end": turn.get(
            "end"
        ),
        "duration": turn.get(
            "duration"
        ),
        "whisper": get_whisper_text(
            turn
        ),
        "shenava": get_shenava_text(
            turn
        ),
    }


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(
    groups: list[dict],
) -> str:

    parts = []

    parts.append(
        "این گروه‌ها بخش متوالی یک مکالمه فارسی هستند."
    )

    parts.append(
        "برای هر GROUP دقیق‌ترین متن نهایی را تولید کن."
    )

    parts.append(
        "گروه‌های مجاور را برای درک context در نظر بگیر."
    )

    parts.append("")

    parts.append(
        "========== INPUT =========="
    )

    parts.append("")

    for group in groups:

        idx = group[
            "group_index"
        ]

        speaker = group[
            "speaker"
        ]

        start = group[
            "start"
        ]

        end = group[
            "end"
        ]

        whisper = group[
            "whisper"
        ]

        shenava = group[
            "shenava"
        ]

        parts.append(
            f"GROUP {idx}"
        )

        parts.append(
            f"SPEAKER: {speaker}"
        )

        parts.append(
            f"TIME: {start} -> {end}"
        )

        parts.append(
            f"WHISPER: {whisper}"
        )

        parts.append(
            f"SHENAVA: {shenava}"
        )

        parts.append("")

    parts.append(
        "========== END INPUT =========="
    )

    parts.append("")

    parts.append(
        "اکنون فقط GROUPها را با متن نهایی تولید کن."
    )

    return "\n".join(parts)


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(
    model: str,
    prompt: str,
    url: str,
) -> str:

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        },
    }

    response = requests.post(
        url,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    message = data.get(
        "message"
    )

    if not isinstance(
        message,
        dict,
    ):
        raise RuntimeError(
            f"پاسخ Ollama فاقد message است:\n{data}"
        )

    content = message.get(
        "content"
    )

    if not content:
        raise RuntimeError(
            f"Ollama متن خروجی برنگرداند:\n{data}"
        )

    return str(
        content
    ).strip()


# ============================================================
# PARSE LLM RESPONSE
# ============================================================

def parse_llm_response(
    response: str,
    expected_indices: list[int],
) -> dict[int, str]:

    result: dict[int, str] = {}

    for raw_line in response.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        # Standard:
        # GROUP 0: text

        match = re.match(
            r"^GROUP\s+(\d+)\s*:\s*(.*)$",
            line,
            flags=re.IGNORECASE,
        )

        if match:

            index = int(
                match.group(1)
            )

            text = clean_text(
                match.group(2)
            )

            if (
                index in expected_indices
                and text
            ):
                result[index] = text

            continue

        # Sometimes model may use:
        #
        # GROUP 0 - text

        match = re.match(
            r"^GROUP\s+(\d+)\s*[-–—]\s*(.*)$",
            line,
            flags=re.IGNORECASE,
        )

        if match:

            index = int(
                match.group(1)
            )

            text = clean_text(
                match.group(2)
            )

            if (
                index in expected_indices
                and text
            ):
                result[index] = text

    return result


# ============================================================
# PROCESS BATCH
# ============================================================

def process_batch(
    batch_number: int,
    groups: list[dict],
    model: str,
    ollama_url: str,
    output_dir: Path,
) -> list[dict]:

    indices = [
        group[
            "group_index"
        ]
        for group in groups
    ]

    print(
        f"[BATCH {batch_number}] "
        f"GROUP {indices[0]} -> GROUP {indices[-1]}"
    )

    prompt = build_prompt(
        groups
    )

    # Save prompt
    prompt_path = (
        output_dir
        / f"batch_{batch_number:04d}_prompt.txt"
    )

    prompt_path.write_text(
        prompt,
        encoding="utf-8",
    )

    started = time.perf_counter()

    try:

        response = call_ollama(
            model=model,
            prompt=prompt,
            url=ollama_url,
        )

    except Exception as exc:

        print(
            f"  ERROR: {exc}"
        )

        error_path = (
            output_dir
            / f"batch_{batch_number:04d}_error.txt"
        )

        error_path.write_text(
            str(exc),
            encoding="utf-8",
        )

        return []

    elapsed = (
        time.perf_counter()
        - started
    )

    # Save raw response
    response_path = (
        output_dir
        / f"batch_{batch_number:04d}_response.txt"
    )

    response_path.write_text(
        response,
        encoding="utf-8",
    )

    parsed = parse_llm_response(
        response,
        indices,
    )

    results = []

    for group in groups:

        index = group[
            "group_index"
        ]

        final_text = parsed.get(
            index,
            "",
        )

        results.append(
            {
                "group_index": index,
                "speaker": group[
                    "speaker"
                ],
                "start": group[
                    "start"
                ],
                "end": group[
                    "end"
                ],
                "whisper": group[
                    "whisper"
                ],
                "shenava": group[
                    "shenava"
                ],
                "final": final_text,
            }
        )

    batch_json = {
        "batch": batch_number,
        "groups": indices,
        "elapsed_seconds": round(
            elapsed,
            3,
        ),
        "results": results,
        "missing_groups": [
            index
            for index in indices
            if index not in parsed
        ],
    }

    save_json(
        output_dir
        / f"batch_{batch_number:04d}.json",
        batch_json,
    )

    print(
        f"  time   : {elapsed:.2f}s"
    )

    print(
        f"  parsed : {len(parsed)}/{len(groups)}"
    )

    if len(parsed) != len(groups):

        missing = [
            index
            for index in indices
            if index not in parsed
        ]

        print(
            f"  missing: {missing}"
        )

    return results


# ============================================================
# FINAL OUTPUT
# ============================================================

def save_final(
    results: list[dict],
    input_path: Path,
    output_dir: Path,
    model: str,
) -> None:

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    final_json = {
        "source_file": str(
            input_path
        ),
        "model": model,
        "groups": results,
    }

    final_json_path = (
        output_dir
        / "final_consensus.json"
    )

    save_json(
        final_json_path,
        final_json,
    )

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    lines = []

    for item in results:

        text = clean_text(
            item.get(
                "final",
                "",
            )
        )

        # LLM fallback
        if not text:

            text = clean_text(
                item.get(
                    "whisper",
                    "",
                )
            )

        if not text:

            text = clean_text(
                item.get(
                    "shenava",
                    "",
                )
            )

        if not text:
            continue

        lines.append(
            "[{} -> {}] {}: {}".format(
                item.get(
                    "start"
                ),
                item.get(
                    "end"
                ),
                item.get(
                    "speaker"
                ),
                text,
            )
        )

    final_txt_path = (
        output_dir
        / f"{input_path.stem}_consensus.txt"
    )

    final_txt_path.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # DEBUG COMPARISON
    # --------------------------------------------------------

    comparison_lines = []

    for item in results:

        comparison_lines.append(
            f"GROUP {item['group_index']}"
        )

        comparison_lines.append(
            f"SPEAKER: {item['speaker']}"
        )

        comparison_lines.append(
            f"WHISPER: {item['whisper']}"
        )

        comparison_lines.append(
            f"SHENAVA: {item['shenava']}"
        )

        comparison_lines.append(
            f"FINAL: {item['final']}"
        )

        comparison_lines.append(
            ""
        )

    comparison_path = (
        output_dir
        / "comparison.txt"
    )

    comparison_path.write_text(
        "\n".join(
            comparison_lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"JSON       : {final_json_path}"
    )

    print(
        f"TXT        : {final_txt_path}"
    )

    print(
        f"Comparison : {comparison_path}"
    )

    print(
        f"Groups     : {len(results)}"
    )

    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Whisper + Shenava consensus "
            "using Qwen 2.5 14B via Ollama"
        )
    )

    parser.add_argument(
        "input_json",
        type=Path,
    )

    parser.add_argument(
        "--model",
        default=MODEL_NAME,
    )

    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_URL,
    )

    parser.add_argument(
        "--groups-per-batch",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    input_path = (
        args.input_json
        .resolve()
    )

    if not input_path.is_file():

        raise FileNotFoundError(
            f"JSON پیدا نشد:\n{input_path}"
        )

    if args.groups_per_batch < 1:

        raise ValueError(
            "--groups-per-batch باید حداقل 1 باشد."
        )

    if args.output_dir:

        output_dir = (
            args.output_dir
            .resolve()
        )

    else:

        output_dir = (
            input_path.parent
            / "llm_consensus"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print("ASR CONSENSUS - QWEN 2.5 14B")
    print("=" * 72)

    print(
        f"Input : {input_path}"
    )

    print(
        f"Model : {args.model}"
    )

    print(
        f"Ollama: {args.ollama_url}"
    )

    print(
        f"Batch : {args.groups_per_batch}"
    )

    print("=" * 72)
    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    data = load_json(
        input_path
    )

    turns = load_turns(
        data
    )

    print(
        f"Loaded turns: {len(turns)}"
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    groups = [
        normalize_turn(turn)
        for turn in turns
    ]

    # Sort by actual group index
    groups.sort(
        key=lambda x: (
            x["group_index"]
            if x["group_index"] is not None
            else 0
        )
    )

    print(
        f"Normalized groups: {len(groups)}"
    )

    print()

    # --------------------------------------------------------
    # TEST OLLAMA
    # --------------------------------------------------------

    print(
        "Testing Ollama..."
    )

    try:

        test = call_ollama(
            model=args.model,
            prompt=(
                "فقط یک کلمه بنویس: آماده"
            ),
            url=args.ollama_url,
        )

    except Exception as exc:

        raise RuntimeError(
            "\n"
            "اتصال به Ollama برقرار نشد.\n"
            f"URL: {args.ollama_url}\n"
            f"Model: {args.model}\n"
            f"Error: {exc}\n"
        ) from exc

    print(
        f"Ollama OK: {test[:100]}"
    )

    print()

    # --------------------------------------------------------
    # BATCH
    # --------------------------------------------------------

    all_results = []

    total_batches = (
        len(groups)
        + args.groups_per_batch
        - 1
    ) // args.groups_per_batch

    for batch_number, start in enumerate(
        range(
            0,
            len(groups),
            args.groups_per_batch,
        ),
        start=1,
    ):

        batch = groups[
            start:
            start + args.groups_per_batch
        ]

        results = process_batch(
            batch_number=batch_number,
            groups=batch,
            model=args.model,
            ollama_url=args.ollama_url,
            output_dir=output_dir,
        )

        all_results.extend(
            results
        )

        print()

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    save_final(
        results=all_results,
        input_path=input_path,
        output_dir=output_dir,
        model=args.model,
    )


if __name__ == "__main__":
    main()