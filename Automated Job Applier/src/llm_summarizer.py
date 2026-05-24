"""Job description summarization with multiple API backends.

Provides concise summaries of job descriptions using:
- Groq (primary)
- Gemini (fallback 1)
- Hugging Face (fallback 2)

Integrated into resume_tailor.py workflow.
"""

from __future__ import annotations

import logging
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

import requests


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _append_api_log(file_name: str, payload: Dict) -> None:
    log_dir = os.getenv("JOB_AUTOMATION_API_LOG_DIR")
    if not log_dir:
        return
    path = Path(log_dir) / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(payload)
    entry.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class SummarizerConfig:
    """Configuration for summarization APIs."""
    groq_api_key_env: str = "GROQ_API_KEY"
    gemini_api_key_env: str = "GEMINI_API_KEY"
    hf_api_key_env: str = "HF_API_KEY"
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.5-flash"
    hf_model: str = "facebook/bart-large-cnn"
    hf_router_base: str = "https://router.huggingface.co/hf-inference/models"
    retry_attempts: int = 3
    retry_delay_seconds: float = 2.0
    max_summary_length: int = 500


def _preview_text(text: str, max_len: int = 220) -> str:
    """Return truncated preview of text for logging."""
    clean = str(text).replace("\n", " ").strip()
    if len(clean) <= max_len:
        return clean
    return clean[:max_len] + "..."


def _retry_call(fn, *args, **kwargs):
    """Retry a function call with exponential backoff."""
    config = SummarizerConfig()
    last_error = None
    
    for attempt in range(1, config.retry_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "%s failed on attempt %s/%s: %s",
                fn.__name__,
                attempt,
                config.retry_attempts,
                exc,
            )
            if attempt < config.retry_attempts:
                time.sleep(config.retry_delay_seconds)
    
    raise RuntimeError(f"All retries failed for {fn.__name__}") from last_error


def _safe_post_json(url: str, headers: Dict[str, str], payload: Dict, timeout: int = 45) -> Dict:
    """POST JSON with HF-specific error handling."""
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code == 403 and "Inference Providers" in response.text:
        raise RuntimeError(
            "Hugging Face token lacks Inference Providers permission. "
            "Enable it in HF settings."
        )
    response.raise_for_status()
    return response.json()


def summarize_with_groq(job_description: str, api_key: str) -> str:
    """Summarize using Groq API."""
    if Groq is None:
        raise ImportError("groq not installed")
    
    config = SummarizerConfig()
    client = Groq(api_key=api_key)
    
    logger.info("Using Groq summarization...")
    prompt = (
        "Provide a summary of this job description highlighting the key responsibilities, skills, requirements etc.:\n\n"
        f"{job_description}"
    )
    
    result = _retry_call(
        client.chat.completions.create,
        model=config.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300,
    )
    
    summary = result.choices[0].message.content.strip() if result.choices else ""
    logger.info("Groq summary: %s", _preview_text(summary))
    _append_api_log(
        "summarizer_calls.jsonl",
        {
            "stage": "summarize_with_groq",
            "provider": "groq",
            "model": config.groq_model,
            "input_chars": len(job_description or ""),
            "summary_preview": _preview_text(summary, 180),
        },
    )
    return summary


def summarize_with_gemini(job_description: str, api_key: str) -> str:
    """Summarize using Gemini API."""
    if genai is None:
        raise ImportError("google.generativeai not installed")
    
    config = SummarizerConfig()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.gemini_model)
    
    logger.info("Using Gemini summarization...")
    prompt = (
        "Provide a summary of this job description highlighting the key responsibilities, skills, requirements etc.:\n\n"
        f"{job_description}"
    )
    
    result = _retry_call(model.generate_content, prompt)
    summary = result.text.strip() if result.text else ""
    logger.info("Gemini summary: %s", _preview_text(summary))
    _append_api_log(
        "summarizer_calls.jsonl",
        {
            "stage": "summarize_with_gemini",
            "provider": "gemini",
            "model": config.gemini_model,
            "input_chars": len(job_description or ""),
            "summary_preview": _preview_text(summary, 180),
        },
    )
    return summary


def summarize_with_hf(job_description: str, api_key: str) -> str:
    """Summarize using Hugging Face API."""
    config = SummarizerConfig()
    url = f"{config.hf_router_base}/{config.hf_model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    logger.info("Using HF summarization...")
    
    # Truncate input if too long for model
    max_input = 1024
    text_to_summarize = job_description[:max_input]
    
    response = _retry_call(
        _safe_post_json,
        url,
        headers,
        {"inputs": text_to_summarize},
    )
    
    if not isinstance(response, list) or not response or "summary_text" not in response[0]:
        raise ValueError("Unexpected HF summarization response format")
    
    summary = response[0]["summary_text"].strip()
    logger.info("HF summary: %s", _preview_text(summary))
    _append_api_log(
        "summarizer_calls.jsonl",
        {
            "stage": "summarize_with_hf",
            "provider": "huggingface",
            "model": config.hf_model,
            "input_chars": len(text_to_summarize or ""),
            "summary_preview": _preview_text(summary, 180),
        },
    )
    return summary


def summarize_job_description(
    job_description: str,
    groq_key: str | None = None,
    gemini_key: str | None = None,
    hf_key: str | None = None,
) -> str:
    """Summarize job description with fallback logic.
    
    Returns:
        Summary text from first successful API call
    """
    # Try Groq first
    if groq_key:
        try:
            summary = summarize_with_groq(job_description, groq_key)
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
            logger.warning("Groq returned empty summary")
        except Exception as exc:
            logger.warning("Groq summarization failed: %s", exc)
            _append_api_log(
                "summarizer_calls.jsonl",
                {
                    "stage": "summarize_job_description",
                    "provider": "groq",
                    "status": "failed",
                    "error": str(exc),
                },
            )
    
    # Try Gemini
    if gemini_key:
        try:
            summary = summarize_with_gemini(job_description, gemini_key)
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
            logger.warning("Gemini returned empty summary")
        except Exception as exc:
            logger.warning("Gemini summarization failed: %s", exc)
            _append_api_log(
                "summarizer_calls.jsonl",
                {
                    "stage": "summarize_job_description",
                    "provider": "gemini",
                    "status": "failed",
                    "error": str(exc),
                },
            )
    
    # Try HF
    if hf_key:
        try:
            summary = summarize_with_hf(job_description, hf_key)
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
            logger.warning("HF returned empty summary")
        except Exception as exc:
            logger.warning("HF summarization failed: %s", exc)
            _append_api_log(
                "summarizer_calls.jsonl",
                {
                    "stage": "summarize_job_description",
                    "provider": "huggingface",
                    "status": "failed",
                    "error": str(exc),
                },
            )
    
    raise RuntimeError(
        "All summarization APIs failed. Check API keys: GROQ_API_KEY, GEMINI_API_KEY, HF_API_KEY"
    )
