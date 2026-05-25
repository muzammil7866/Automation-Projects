"""Embedding-based job matcher with multiple API backends for similarity scoring.

Uses semantic embeddings to compute resume-job description similarity.
Provides 3 API backends (Gemini, HF primary, HF alt) with fallback logic.

Run via main matcher.py with --use-embeddings flag or integrated directly.
"""

from __future__ import annotations

import logging
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    genai = None


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
class EmbedderConfig:
    """Configuration for embedding-based matching."""
    gemini_api_key_env: str = "GEMINI_API_KEY"
    hf_api_key_env: str = "HF_API_KEY"
    embedding_model_gemini: str = "models/gemini-embedding-001"
    embedding_model_hf: str = "BAAI/bge-small-en-v1.5"
    embedding_model_hf_alt: str = "BAAI/bge-base-en-v1.5"
    hf_router_base: str = "https://router.huggingface.co/hf-inference/models"
    retry_attempts: int = 3
    retry_delay_seconds: float = 2.0


def _preview_vector(vector: np.ndarray, max_len: int = 5) -> List[float]:
    """Return first N elements of vector for logging."""
    if isinstance(vector, list):
        vector = np.array(vector)
    return vector[:max_len].tolist()


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


def _retry_call(fn, *args, **kwargs):
    """Retry a function call with exponential backoff."""
    config = EmbedderConfig()
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


def embed_with_gemini(text: str, api_key: str) -> np.ndarray:
    """Embed text using Google Gemini embedding API."""
    if genai is None:
        raise ImportError("google.generativeai not installed")
    
    config = EmbedderConfig()
    genai.configure(api_key=api_key)
    
    logger.info("Using Gemini embedding...")
    result = _retry_call(
        genai.embed_content,
        model=config.embedding_model_gemini,
        content=text,
    )
    embedding = np.array(result["embedding"])
    logger.info("Gemini embedding dim=%s preview=%s", embedding.size, _preview_vector(embedding))
    _append_api_log(
        "matcher_embeddings.jsonl",
        {
            "stage": "embed_with_gemini",
            "provider": "gemini",
            "model": config.embedding_model_gemini,
            "input_chars": len(text),
            "embedding_dim": int(embedding.size),
        },
    )
    return embedding


def embed_with_hf(text: str, api_key: str, model_name: str = "default") -> np.ndarray:
    """Embed text using Hugging Face API."""
    config = EmbedderConfig()
    model = config.embedding_model_hf if model_name == "default" else config.embedding_model_hf_alt
    
    url = f"{config.hf_router_base}/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    logger.info("Using HF embedding (%s)...", model)
    response = _retry_call(_safe_post_json, url, headers, {"inputs": text})
    
    embedding = np.array(response)
    logger.info("HF embedding dim=%s preview=%s", embedding.size, _preview_vector(embedding))
    _append_api_log(
        "matcher_embeddings.jsonl",
        {
            "stage": "embed_with_hf",
            "provider": "huggingface",
            "model": model,
            "input_chars": len(text),
            "embedding_dim": int(embedding.size),
        },
    )
    return embedding


def embed_with_hf_alt(text: str, api_key: str) -> np.ndarray:
    """Embed text using alternative Hugging Face model."""
    return embed_with_hf(text, api_key, model_name="alt")


def compute_embedding(
    text: str,
    gemini_key: str | None = None,
    hf_key: str | None = None,
) -> Tuple[np.ndarray, str]:
    """Compute embedding with primary and fallback APIs.
    
    Returns:
        (embedding_vector, source_api_name)
    """
    # Try Gemini first
    if gemini_key:
        try:
            embedding = embed_with_gemini(text, gemini_key)
            if embedding is not None and len(embedding) > 0:
                return embedding, "gemini"
        except Exception as exc:
            logger.warning("Gemini embedding failed: %s", exc)
            _append_api_log(
                "matcher_embeddings.jsonl",
                {
                    "stage": "compute_embedding",
                    "provider": "gemini",
                    "status": "failed",
                    "error": str(exc),
                },
            )
    
    # Try HF primary
    if hf_key:
        try:
            embedding = embed_with_hf(text, hf_key, model_name="default")
            if embedding is not None and len(embedding) > 0:
                return embedding, "hf_primary"
        except Exception as exc:
            logger.warning("HF primary embedding failed: %s", exc)
            _append_api_log(
                "matcher_embeddings.jsonl",
                {
                    "stage": "compute_embedding",
                    "provider": "hf_primary",
                    "status": "failed",
                    "error": str(exc),
                },
            )
    
    # Try HF alt
    if hf_key:
        try:
            embedding = embed_with_hf_alt(text, hf_key)
            if embedding is not None and len(embedding) > 0:
                return embedding, "hf_alt"
        except Exception as exc:
            logger.warning("HF alt embedding failed: %s", exc)
            _append_api_log(
                "matcher_embeddings.jsonl",
                {
                    "stage": "compute_embedding",
                    "provider": "hf_alt",
                    "status": "failed",
                    "error": str(exc),
                },
            )
    
    raise RuntimeError(
        "All embedding APIs failed. Check API keys: GEMINI_API_KEY, HF_API_KEY"
    )


def compute_similarity_score(
    resume_embedding: np.ndarray,
    job_description: str,
    gemini_key: str | None = None,
    hf_key: str | None = None,
) -> Tuple[int, str]:
    """Compute cosine similarity between resume and job description.
    
    Returns:
        (similarity_score_0_to_100, source_api_name)
    """
    job_embedding, source = compute_embedding(job_description, gemini_key, hf_key)
    
    # Handle dimension mismatch by resizing to smaller dimension
    if resume_embedding.shape != job_embedding.shape:
        min_dim = min(resume_embedding.shape[0], job_embedding.shape[0])
        resume_embedding = resume_embedding[:min_dim]
        job_embedding = job_embedding[:min_dim]
    
    similarity = 1 - cosine(resume_embedding, job_embedding)
    score = round(similarity * 100)
    score = max(0, min(100, score))
    
    logger.info("Similarity (from %s): %.2f%%", source, score)
    _append_api_log(
        "matcher_similarity.jsonl",
        {
            "stage": "compute_similarity_score",
            "provider": source,
            "score": int(score),
            "job_description_chars": len(job_description or ""),
        },
    )
    return score, source


def compute_resume_embedding(
    resume_text: str,
    gemini_key: str | None = None,
    hf_key: str | None = None,
) -> np.ndarray:
    """Compute and cache resume embedding."""
    embedding, source = compute_embedding(resume_text, gemini_key, hf_key)
    logger.info("Resume embedded using %s", source)
    return embedding
