"""Two-step API-based resume tailoring for matched jobs.

Workflow:
1) Load top jobs from data/filtered_jobs.json
2) Extract JD signals (top keywords + role summary + stack + requirements)
3) Generate tailored LaTeX from source .tex and extracted signals
4) Save per-job outputs under output/tailored_resumes/
5) Optionally compile PDF with pdflatex

Default hybrid setup:
- Step 0 (Optional): Summarize job description using multiple APIs with fallback
- Step 1 (Extraction): OpenRouter, SiliconFlow, Groq, or Custom API
- Step 2 (Generation): OpenRouter, SiliconFlow, Groq, or Custom API

All steps support retry logic and graceful fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

try:
    from llm_summarizer import summarize_job_description
    HAS_LLM_SUMMARIZER = True
except ImportError:
    HAS_LLM_SUMMARIZER = False


@dataclass(frozen=True)
class TailorConfig:
    filtered_jobs_path: str = "data/filtered_jobs.json"
    output_dir: str = "output/tailored_resumes"
    use_summarization: bool = False
    summarizer_groq_api_key_env: str = "GROQ_API_KEY"
    summarizer_gemini_api_key_env: str = "GEMINI_API_KEY"
    summarizer_hf_api_key_env: str = "HF_API_KEY"
    extract_provider: str = "groq"
    extract_model_name: str = "llama-3.3-70b-versatile"
    extract_base_url: str = "https://api.groq.com/openai/v1"
    extract_api_key_env: str = "GROQ_API_KEY"
    generate_provider: str = "groq"
    generate_model_name: str = "llama-3.3-70b-versatile"
    generate_base_url: str = "https://api.groq.com/openai/v1"
    generate_api_key_env: str = "GROQ_API_KEY"
    max_jobs: int = 3
    min_match_score: int = 0
    retries: int = 3
    temperature: float = 0.2
    compile_pdf: bool = False
    latex_engine: str = "pdflatex"
    latex_timeout_sec: int = 60


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_project_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9\-\s_]+", "", value)
    value = re.sub(r"[\s_]+", "-", value.strip().lower())
    return value[:80] or "job"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_api_log(file_name: str, payload: Dict[str, Any]) -> None:
    log_dir = os.getenv("JOB_AUTOMATION_API_LOG_DIR")
    if not log_dir:
        return
    path = Path(log_dir) / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(payload)
    entry.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract first JSON object from model output robustly."""
    if not text:
        raise ValueError("Empty model response")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse JSON object from model response")


def _resolve_provider_settings(provider: str, model: str | None, base_url: str | None, api_key_env: str | None) -> Tuple[str, str, str]:
    provider = (provider or "openrouter").strip().lower()

    defaults = {
        "openrouter": {
            "model": "qwen/qwen-turbo",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        "siliconflow": {
            "model": "deepseek-ai/DeepSeek-V3",
            "base_url": "https://api.siliconflow.cn/v1",
            "api_key_env": "SILICONFLOW_API_KEY",
        },
        "groq": {
            "model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key_env": "GROQ_API_KEY",
        },
        "custom": {
            "model": "deepseek/deepseek-chat",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "LLM_API_KEY",
        },
    }

    if provider not in defaults:
        raise ValueError(f"Unsupported provider: {provider}")

    selected = defaults[provider]
    final_model = model or selected["model"]
    final_base_url = base_url or selected["base_url"]
    final_api_key_env = api_key_env or selected["api_key_env"]
    return final_model, final_base_url, final_api_key_env


def _resolve_extraction_settings(provider: str, model: str | None, base_url: str | None, api_key_env: str | None) -> Tuple[str, str, str, str]:
    provider = (provider or "openrouter").strip().lower()
    defaults = {
        "openrouter": {
            "model": "qwen/qwen-plus",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        "siliconflow": {
            "model": "deepseek-ai/DeepSeek-V3",
            "base_url": "https://api.siliconflow.cn/v1",
            "api_key_env": "SILICONFLOW_API_KEY",
        },
        "groq": {
            "model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key_env": "GROQ_API_KEY",
        },
        "custom": {
            "model": "deepseek/deepseek-chat",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "LLM_API_KEY",
        },
    }
    if provider not in defaults:
        raise ValueError(f"Unsupported extraction provider: {provider}")
    selected = defaults[provider]
    final_model = model or selected["model"]
    final_base_url = base_url or selected["base_url"]
    final_api_key_env = api_key_env or selected["api_key_env"]
    return provider, final_model, final_base_url, final_api_key_env


def _call_llm_json_with_retry(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    retries: int,
    temperature: float,
    call_tag: str,
) -> Dict[str, Any]:
    delay = 1.5
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = completion.choices[0].message.content if completion.choices else ""
            parsed = _extract_json_object(content or "")
            _append_api_log(
                "tailor_extract_calls.jsonl",
                {
                    "stage": call_tag,
                    "model": model_name,
                    "temperature": temperature,
                    "request_chars": len(user_prompt),
                    "response_chars": len(content or ""),
                    "status": "success",
                },
            )
            return parsed
        except Exception as exc:  # pragma: no cover - network/runtime behavior
            last_error = exc
            _append_api_log(
                "tailor_extract_calls.jsonl",
                {
                    "stage": call_tag,
                    "model": model_name,
                    "temperature": temperature,
                    "request_chars": len(user_prompt),
                    "status": "failed",
                    "attempt": attempt,
                    "error": str(exc),
                },
            )
            if attempt == retries:
                break
            jitter = random.uniform(0.0, 0.8)
            time.sleep(delay + jitter)
            delay *= 2

    raise RuntimeError(f"LLM JSON call failed after {retries} attempts: {last_error}")


def _call_llm_text_with_retry(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    retries: int,
    temperature: float,
    call_tag: str,
) -> str:
    delay = 1.5
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = completion.choices[0].message.content if completion.choices else ""
            if content and content.strip():
                _append_api_log(
                    "tailor_generate_calls.jsonl",
                    {
                        "stage": call_tag,
                        "model": model_name,
                        "temperature": temperature,
                        "request_chars": len(user_prompt),
                        "response_chars": len(content),
                        "status": "success",
                    },
                )
                return content.strip()
            raise RuntimeError("LLM text response was empty")
        except Exception as exc:  # pragma: no cover - network/runtime behavior
            last_error = exc
            _append_api_log(
                "tailor_generate_calls.jsonl",
                {
                    "stage": call_tag,
                    "model": model_name,
                    "temperature": temperature,
                    "request_chars": len(user_prompt),
                    "status": "failed",
                    "attempt": attempt,
                    "error": str(exc),
                },
            )
            if attempt == retries:
                break
            jitter = random.uniform(0.0, 0.8)
            time.sleep(delay + jitter)
            delay *= 2

    raise RuntimeError(f"LLM text call failed after {retries} attempts: {last_error}")



def extract_job_focus(client: OpenAI, job: Dict[str, Any], config: TailorConfig) -> Dict[str, Any]:
    system_prompt = (
        "You extract structured hiring signals from job descriptions. "
        "Return strict JSON only."
    )
    user_prompt = f"""
Extract resume-tailoring targets from this job.

Return STRICT JSON with this exact schema:
{{
  "target_title": "string",
  "top_keywords": ["string"],
  "role_summary": "string",
  "tech_stack": ["string"],
  "core_requirements": ["string"]
}}

Rules:
- top_keywords must contain exactly 10 short terms.
- role_summary must be exactly one sentence.
- core_requirements should be concise and concrete.
- Do not include markdown or extra text.

JOB TITLE: {job.get("job_title", "")}
COMPANY: {job.get("company", "")}
LOCATION: {job.get("location", "")}
JOB DESCRIPTION:
{job.get("job_description", "")}
""".strip()

    focus = _call_llm_json_with_retry(
        client=client,
        model_name=config.extract_model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        retries=config.retries,
        temperature=config.temperature,
        call_tag="extract_job_focus",
    )

    focus.setdefault("target_title", str(job.get("job_title", "")))
    focus.setdefault("top_keywords", [])
    focus.setdefault("role_summary", "")
    focus.setdefault("tech_stack", [])
    focus.setdefault("core_requirements", [])

    keywords = [str(k).strip() for k in focus.get("top_keywords", []) if str(k).strip()]
    if len(keywords) < 10:
        fallback = [str(k).strip() for k in focus.get("tech_stack", []) if str(k).strip()]
        keywords.extend(fallback)
    focus["top_keywords"] = keywords[:10]

    role_summary = str(focus.get("role_summary", "")).strip()
    if role_summary and not role_summary.endswith((".", "!", "?")):
        role_summary = f"{role_summary}."
    focus["role_summary"] = role_summary
    return focus


def tailor_resume_tex(
    client: OpenAI,
    resume_tex: str,
    job: Dict[str, Any],
    focus: Dict[str, Any],
    config: TailorConfig,
) -> str:
    system_prompt = (
        "You are an expert ATS-focused resume writer and LaTeX editor. "
        "Output only valid LaTeX, with no markdown fencing and no explanations."
    )
    user_prompt = f"""
You will tailor a LaTeX resume for this role.

Required edits:
1) Integrate the provided top keywords naturally into existing experience bullet points.

Strict constraints:
- Output ONLY valid LaTeX code.
- Do NOT change the document class or any preamble content.
- Keep all facts truthful; do not invent projects, employers, tools, dates, or metrics.
- Preserve existing section structure unless minimal wording changes are needed.

ROLE SUMMARY:
{focus.get("role_summary", "")}

TOP KEYWORDS:
{json.dumps(focus.get("top_keywords", []), ensure_ascii=False)}

TECH STACK:
{json.dumps(focus.get("tech_stack", []), ensure_ascii=False)}

CORE REQUIREMENTS:
{json.dumps(focus.get("core_requirements", []), ensure_ascii=False)}

TARGET JOB TITLE: {job.get("job_title", "")}
COMPANY: {job.get("company", "")}

SOURCE RESUME LATEX:
{resume_tex}
""".strip()

    return _call_llm_text_with_retry(
        client=client,
        model_name=config.generate_model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        retries=config.retries,
        temperature=config.temperature,
        call_tag="tailor_resume_tex",
    )


def _maybe_summarize_job(
    job: Dict[str, Any],
    config: TailorConfig,
    logger: logging.Logger,
) -> str:
    """Optionally summarize job description using multi-provider fallback."""
    if not config.use_summarization or not HAS_LLM_SUMMARIZER:
        _append_api_log(
            "summarizer_calls.jsonl",
            {
                "stage": "resume_tailor_precheck",
                "status": "skipped",
                "reason": "disabled_or_module_missing",
                "job_title": str(job.get("job_title", "")),
            },
        )
        return job.get("job_description", "")
    
    try:
        load_dotenv()
        groq_key = os.getenv(config.summarizer_groq_api_key_env)
        gemini_key = os.getenv(config.summarizer_gemini_api_key_env)
        hf_key = os.getenv(config.summarizer_hf_api_key_env)
        
        if not any([groq_key, gemini_key, hf_key]):
            logger.warning(
                "Summarization enabled but no API keys found. Check %s, %s, %s env vars.",
                config.summarizer_groq_api_key_env,
                config.summarizer_gemini_api_key_env,
                config.summarizer_hf_api_key_env,
            )
            _append_api_log(
                "summarizer_calls.jsonl",
                {
                    "stage": "resume_tailor_precheck",
                    "status": "skipped",
                    "reason": "no_api_keys",
                    "job_title": str(job.get("job_title", "")),
                },
            )
            return job.get("job_description", "")
        
        job_description = job.get("job_description", "")
        summary = summarize_job_description(job_description, groq_key, gemini_key, hf_key)
        logger.info("Summarized job description for %s", job.get("job_title", "unknown"))
        return summary
    except Exception as exc:
        logger.warning("Job summarization failed: %s. Using original description.", exc)
        _append_api_log(
            "summarizer_calls.jsonl",
            {
                "stage": "resume_tailor_precheck",
                "status": "failed",
                "reason": "fallback_to_original",
                "job_title": str(job.get("job_title", "")),
                "error": str(exc),
            },
        )
        return job.get("job_description", "")


def _split_tex(tex: str) -> Tuple[str, str] | None:
    begin_token = "\\begin{document}"
    end_token = "\\end{document}"
    start = tex.find(begin_token)
    end = tex.rfind(end_token)
    if start < 0 or end < 0 or end <= start:
        return None
    preamble = tex[: start + len(begin_token)]
    body = tex[start + len(begin_token) : end]
    return preamble, body


def _validate_tailored_tex(tex: str, source_tex: str) -> str:
    """Basic LaTeX sanity checks + enforced source preamble with safe fallback."""
    candidate = (tex or "").strip()
    if not candidate:
        return source_tex

    must_contain = ["\\begin{document}", "\\end{document}"]
    if not all(token in candidate for token in must_contain):
        return source_tex

    # Avoid accidental truncation from model output.
    if len(candidate) < max(500, int(0.35 * len(source_tex))):
        return source_tex

    source_parts = _split_tex(source_tex)
    candidate_parts = _split_tex(candidate)
    if not source_parts or not candidate_parts:
        return source_tex

    source_preamble, _ = source_parts
    _, candidate_body = candidate_parts
    rebuilt = f"{source_preamble}{candidate_body}\\end{{document}}"
    return rebuilt


def _compile_pdf(tex_path: Path, engine: str, timeout_sec: int) -> Tuple[bool, str]:
    command = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(tex_path.parent),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - environment/runtime behavior
        return False, f"Compile failed to execute: {exc}"

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    pdf_exists = tex_path.with_suffix(".pdf").exists()
    if completed.returncode == 0 and pdf_exists:
        return True, "PDF compiled successfully"
    return False, output[-3000:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tailor LaTeX resume for matched jobs using a two-step LLM workflow")
    parser.add_argument("--resume-tex-file", required=True, help="Path to source LaTeX resume (.tex)")
    parser.add_argument("--filtered-jobs-path", default="data/filtered_jobs.json", help="Path to filtered jobs JSON")
    parser.add_argument("--output-dir", default="output/tailored_resumes", help="Output directory for tailored resumes")
    parser.add_argument("--max-jobs", type=int, default=3, help="Max number of jobs to tailor per run")
    parser.add_argument("--min-match-score", type=int, default=0, help="Minimum match_score to process")
    parser.add_argument(
        "--use-summarization",
        action="store_true",
        help="Summarize job descriptions before extraction (uses multiple APIs with fallback)",
    )
    parser.add_argument(
        "--summarizer-groq-api-key-env",
        default="GROQ_API_KEY",
        help="Env var for Groq API key (summarization)",
    )
    parser.add_argument(
        "--summarizer-gemini-api-key-env",
        default="GEMINI_API_KEY",
        help="Env var for Gemini API key (summarization)",
    )
    parser.add_argument(
        "--summarizer-hf-api-key-env",
        default="HF_API_KEY",
        help="Env var for HF API key (summarization)",
    )
    parser.add_argument(
        "--extract-provider",
        choices=["openrouter", "siliconflow", "groq", "custom"],
        default="groq",
        help="Provider for extraction step",
    )
    parser.add_argument("--extract-model", default=None, help="Extraction model name")
    parser.add_argument("--extract-base-url", default=None, help="Override extraction base URL")
    parser.add_argument("--extract-api-key-env", default=None, help="Env var name for extraction API key")
    parser.add_argument(
        "--generate-provider",
        choices=["openrouter", "siliconflow", "groq", "custom"],
        default="groq",
        help="Provider for generation step",
    )
    parser.add_argument("--generate-model", default=None, help="Generation model name")
    parser.add_argument("--generate-base-url", default=None, help="Override generation base URL")
    parser.add_argument("--generate-api-key-env", default=None, help="Env var name for generation API key")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts for API calls")
    parser.add_argument("--compile-pdf", action="store_true", help="Compile each generated .tex to PDF using pdflatex")
    parser.add_argument("--latex-engine", default="pdflatex", help="Local LaTeX compiler command")
    parser.add_argument("--latex-timeout-sec", type=int, default=60, help="Timeout for each LaTeX compile")
    return parser.parse_args()


def run_resume_tailor() -> List[Dict[str, Any]]:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    args = parse_args()

    extract_provider, extract_model_name, extract_base_url, extract_api_key_env = _resolve_extraction_settings(
        provider=args.extract_provider,
        model=args.extract_model,
        base_url=args.extract_base_url,
        api_key_env=args.extract_api_key_env,
    )
    generate_model_name, generate_base_url, generate_api_key_env = _resolve_provider_settings(
        provider=args.generate_provider,
        model=args.generate_model,
        base_url=args.generate_base_url,
        api_key_env=args.generate_api_key_env,
    )

    config = TailorConfig(
        filtered_jobs_path=args.filtered_jobs_path,
        output_dir=args.output_dir,
        use_summarization=args.use_summarization,
        summarizer_groq_api_key_env=args.summarizer_groq_api_key_env,
        summarizer_gemini_api_key_env=args.summarizer_gemini_api_key_env,
        summarizer_hf_api_key_env=args.summarizer_hf_api_key_env,
        extract_provider=extract_provider,
        extract_model_name=extract_model_name,
        extract_base_url=extract_base_url,
        extract_api_key_env=extract_api_key_env,
        generate_provider=args.generate_provider,
        generate_model_name=generate_model_name,
        generate_base_url=generate_base_url,
        generate_api_key_env=generate_api_key_env,
        max_jobs=args.max_jobs,
        min_match_score=args.min_match_score,
        retries=args.retries,
        temperature=args.temperature,
        compile_pdf=args.compile_pdf,
        latex_engine=args.latex_engine,
        latex_timeout_sec=args.latex_timeout_sec,
    )

    load_dotenv()
    extract_api_key = os.getenv(config.extract_api_key_env)
    if not extract_api_key:
        raise EnvironmentError(
            f"{config.extract_api_key_env} is missing. Add it to environment or .env file."
        )
    generate_api_key = os.getenv(config.generate_api_key_env)
    if not generate_api_key:
        raise EnvironmentError(
            f"{config.generate_api_key_env} is missing. Add it to environment or .env file."
        )

    extract_client = OpenAI(api_key=extract_api_key, base_url=config.extract_base_url)
    generate_client = OpenAI(api_key=generate_api_key, base_url=config.generate_base_url)

    resume_tex_path = _resolve_project_path(args.resume_tex_file)
    if not resume_tex_path.exists():
        raise FileNotFoundError(f"resume tex file not found: {resume_tex_path}")
    source_resume_tex = resume_tex_path.read_text(encoding="utf-8", errors="ignore")

    filtered_jobs_path = _resolve_project_path(config.filtered_jobs_path)
    if not filtered_jobs_path.exists():
        raise FileNotFoundError(f"filtered jobs file not found: {filtered_jobs_path}")

    jobs_payload = _read_json(filtered_jobs_path)
    if not isinstance(jobs_payload, list):
        raise ValueError("filtered_jobs.json must contain a list")

    selected_jobs = [
        job for job in jobs_payload if isinstance(job, dict) and int(job.get("match_score", 0)) >= config.min_match_score
    ][: max(0, config.max_jobs)]

    output_base = _resolve_project_path(config.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for idx, job in enumerate(selected_jobs, start=1):
        title = str(job.get("job_title", "job"))
        company = str(job.get("company", "company"))
        job_slug = _slugify(f"{idx:02d}-{title}-{company}")
        job_dir = output_base / job_slug
        job_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Tailoring resume for job {idx}/{len(selected_jobs)}: {title} @ {company}")

        # Step 0: Optional summarization
        job_description = _maybe_summarize_job(job, config, logger)
        job_with_summary = dict(job)
        job_with_summary["job_description"] = job_description
        
        # Step 1: Extract job focus
        focus = extract_job_focus(extract_client, job_with_summary, config)
        # Step 2: Generate tailored LaTeX
        generated_tex = tailor_resume_tex(generate_client, source_resume_tex, job, focus, config)
        tailored_tex = _validate_tailored_tex(generated_tex, source_resume_tex)

        tailored_tex_path = job_dir / "resume_tailored.tex"
        tailored_tex_path.write_text(tailored_tex, encoding="utf-8")

        compile_result = {
            "enabled": config.compile_pdf,
            "success": False,
            "message": "PDF compile skipped",
            "pdf_path": "",
        }
        if config.compile_pdf:
            ok, message = _compile_pdf(
                tex_path=tailored_tex_path,
                engine=config.latex_engine,
                timeout_sec=config.latex_timeout_sec,
            )
            compile_result = {
                "enabled": True,
                "success": ok,
                "message": message,
                "pdf_path": str(tailored_tex_path.with_suffix(".pdf")) if ok else "",
            }

        summary_payload = {
            "job": {
                "job_title": job.get("job_title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "match_score": job.get("match_score", 0),
                "apply_link": job.get("apply_link", ""),
            },
            "focus": focus,
            "tailored_title": focus.get("target_title", title),
            "output_tex": str(tailored_tex_path),
            "compile": compile_result,
        }

        _write_json(job_dir / "tailor_summary.json", summary_payload)
        results.append(summary_payload)

    _write_json(output_base / "run_summary.json", {"count": len(results), "items": results})
    print(f"[INFO] Summarization: {'enabled' if config.use_summarization else 'disabled'}")
    print(f"[INFO] Extract: {config.extract_provider} | Model: {config.extract_model_name}")
    print(f"[INFO] Generate: {config.generate_provider} | Model: {config.generate_model_name}")
    print(f"[INFO] Tailored resumes created: {len(results)}")
    print(f"[INFO] Output directory: {output_base}")
    return results


if __name__ == "__main__":
    run_resume_tailor()
