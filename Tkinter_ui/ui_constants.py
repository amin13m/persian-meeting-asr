from __future__ import annotations

import os
import sys
from pathlib import Path

# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Offline Mode
# ============================================================

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ============================================================
# Internal Imports
# ============================================================

from engines.whisper_persian_v4.engine import WhisperPersianV4Engine

# ============================================================
# Paths
# ============================================================

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "input"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
RECORDINGS_DIR = DEFAULT_INPUT_DIR / "recordings"

DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Engine Registry
# ============================================================

ENGINE_REGISTRY = {
    "whisper_persian_v4": WhisperPersianV4Engine,
}

TEXTS = {
    "en": {
        "app_title": "Persian AI Speech-to-Text",
        "app_subtitle": "Local • Private • Offline Persian Speech Recognition",

        "main": "Main",
        "logs": "Logs",

        "model_section": "Model",
        "model": "Model",
        "model_status": "Model Status",
        "load_model": "Load Model",
        "unload_model": "Unload Model",

        "audio_section": "Audio",
        "audio_file": "Audio File",
        "select_audio": "Select Audio",
        "record": "Recording",
        "start_recording": "Start Recording",
        "stop_recording": "Stop Recording",

        "processing_section": "Processing",
        "transcribe": "Transcribe",

        "transcription_section": "Transcription",
        "no_transcription": "No transcription yet.",

        "output_section": "Export",
        "export_txt": "Export TXT",
        "export_json": "Export JSON",
        "open_output": "Open Output",

        "settings": "Settings",
        "language": "Language",
        "theme": "Theme",
        "english": "English",
        "persian": "فارسی",
        "dark": "Dark",
        "light": "Light",

        "status": "Status",
        "ready": "Ready",
        "loading": "Loading model...",
        "loaded": "Model loaded",
        "unloading": "Unloading model...",
        "unloaded": "Model unloaded",
        "processing": "Processing...",
        "recording": "Recording...",
        "completed": "Completed",
        "error": "Error",
        "recording_saved": "Recording saved",

        "select_audio_first": "Please select an audio file first.",
        "load_model_first": "Please load the model first.",
        "model_load_failed": "Model loading failed.",
        "transcription_failed": "Transcription failed.",
        "select_model": "Select model",

        "chunks": "Chunks",
        "chunk": "Chunk",
        "processing_chunk": "Processing chunk",
        "elapsed": "Elapsed",

        "logs_title": "Application Logs",
        "clear_logs": "Clear Logs",

        "open_audio": "Select Audio File",
        "audio_files": "Audio Files",
        "all_files": "All Files",

        "success": "Success",
        "warning": "Warning",
        "info": "Info",

        "saved_txt": "TXT file saved:",
        "saved_json": "JSON file saved:",
        
        "copy_text": "Copy Text",


        "confirm_close_title": "Close Application",
        "confirm_close": "An operation is still running. Close application anyway?",
    },

    "fa": {
        "app_title": "تبدیل گفتار فارسی به متن",
        "app_subtitle": "محلی • خصوصی • آفلاین • تشخیص گفتار فارسی",

        "main": "اصلی",
        "logs": "لاگ‌ها",

        "model_section": "مدل",
        "model": "مدل",
        "model_status": "وضعیت مدل",
        "load_model": "بارگذاری مدل",
        "unload_model": "خارج کردن مدل",

        "audio_section": "صوت",
        "audio_file": "فایل صوتی",
        "select_audio": "انتخاب فایل",
        "record": "ضبط صدا",
        "start_recording": "شروع ضبط",
        "stop_recording": "توقف ضبط",

        "processing_section": "پردازش",
        "transcribe": "تبدیل به متن",

        "transcription_section": "متن خروجی",
        "no_transcription": "هنوز متنی تولید نشده است.",

        "output_section": "خروجی",
        "export_txt": "خروجی TXT",
        "export_json": "خروجی JSON",
        "open_output": "باز کردن پوشه خروجی",

        "settings": "تنظیمات",
        "language": "زبان",
        "theme": "پوسته",
        "english": "English",
        "persian": "فارسی",
        "dark": "تیره",
        "light": "روشن",

        "status": "وضعیت",
        "ready": "آماده",
        "loading": "در حال بارگذاری مدل...",
        "loaded": "مدل آماده است",
        "unloading": "در حال آزادسازی مدل...",
        "unloaded": "مدل آزاد شد",
        "processing": "در حال پردازش...",
        "recording": "در حال ضبط...",
        "completed": "با موفقیت انجام شد",
        "error": "خطا",
        "recording_saved": "ضبط ذخیره شد",

        "select_audio_first": "ابتدا یک فایل صوتی انتخاب کنید.",
        "load_model_first": "ابتدا مدل را بارگذاری کنید.",
        "model_load_failed": "بارگذاری مدل با خطا مواجه شد.",
        "transcription_failed": "تبدیل گفتار به متن با خطا مواجه شد.",
        "select_model": "مدل را انتخاب کنید",

        "chunks": "بخش",
        "chunk": "بخش",
        "processing_chunk": "در حال پردازش بخش",
        "elapsed": "زمان",

        "logs_title": "لاگ اجرای برنامه",
        "clear_logs": "پاک کردن لاگ‌ها",

        "open_audio": "انتخاب فایل صوتی",
        "audio_files": "فایل‌های صوتی",
        "all_files": "همه فایل‌ها",

        "success": "موفق",
        "warning": "هشدار",
        "info": "اطلاعات",

        "saved_txt": "فایل TXT ذخیره شد:",
        "saved_json": "فایل JSON ذخیره شد:",
        
        "copy_text": "کپی متن",

        "confirm_close_title": "بستن برنامه",
        "confirm_close": "یک عملیات هنوز در حال اجرا است. برنامه بسته شود؟",
    },
}


# ============================================================
# Themes
# ============================================================

THEMES = {
    "dark": {
        "bg": "#17191D",
        "surface": "#202328",
        "surface2": "#272B31",
        "border": "#343942",
        "text": "#ECEFF4",
        "muted": "#9AA3AE",
        "accent": "#6EA8FE",
        "accent_hover": "#82B5FF",
        "accent_pressed": "#4F8FE8",
        "success": "#66C28A",
        "warning": "#E6B566",
        "error": "#E27D7D",
        "info": "#78A8D8",
        "disabled": "#50555D",
        "log_bg": "#15171A",
        "input_bg": "#1C1F24",
        "selection": "#315C91",
    },

    "light": {
        "bg": "#F3F5F7",
        "surface": "#FFFFFF",
        "surface2": "#F8F9FA",
        "border": "#D9DEE5",
        "text": "#20242A",
        "muted": "#69727D",
        "accent": "#4F82D8",
        "accent_hover": "#5F91E3",
        "accent_pressed": "#3D70C5",
        "success": "#3C9B68",
        "warning": "#B47A28",
        "error": "#C95757",
        "info": "#477CA9",
        "disabled": "#A7ADB5",
        "log_bg": "#F7F8FA",
        "input_bg": "#FFFFFF",
        "selection": "#C9DDF8",
    },
}


# ============================================================
# Helpers
# ============================================================

def is_persian_language(language: str) -> bool:
    return language == "fa"


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

    return f"{minutes:02d}:{secs:05.2f}"