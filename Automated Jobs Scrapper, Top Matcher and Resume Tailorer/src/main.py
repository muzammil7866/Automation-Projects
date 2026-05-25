"""Main entrypoint for running the job automation MVP pipeline.

Pipeline:
1) Scrape jobs -> data/jobs.json
2) Match jobs against resume -> data/filtered_jobs.json
3) (Optional) Tailor LaTeX resume via two-step LLM API -> output/tailored_resumes/

Run:
    python src/main.py
    python src/main.py --resume-text-file data/resume_match.txt --top-n 15
    python src/main.py --run-tailor --resume-tex-file data/resume.tex
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from scraper import run_scraper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scraper + matcher pipeline in one command")
    parser.add_argument(
        "--resume-text-file",
        default="",
        help="Optional resume text path used by matcher (.txt/.tex/.md)",
    )
    parser.add_argument("--top-n", type=int, default=15, help="Number of top matched jobs to save")
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Enable embedding-based semantic matching in matcher",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping and run matcher on existing data/jobs.json",
    )
    parser.add_argument(
        "--run-tailor",
        action="store_true",
        help="Run two-step resume tailoring after matcher",
    )
    parser.add_argument(
        "--resume-tex-file",
        default=None,
        help="Path to source LaTeX resume for tailoring (required with --run-tailor)",
    )
    parser.add_argument(
        "--tailor-max-jobs",
        type=int,
        default=3,
        help="Max number of matched jobs to tailor",
    )
    parser.add_argument(
        "--tailor-min-match-score",
        type=int,
        default=0,
        help="Only tailor jobs with at least this match score",
    )
    parser.add_argument(
        "--tailor-output-dir",
        default="output/tailored_resumes",
        help="Output directory for tailored resume artifacts",
    )
    parser.add_argument(
        "--tailor-extract-provider",
        choices=["openrouter", "siliconflow", "groq", "custom"],
        default="custom",
        help="Extraction provider for Step 1 (default: 'custom' — set explicitly to run tailoring)",
    )
    parser.add_argument(
        "--tailor-extract-model",
        default=None,
        help="Extraction model for Step 1 (default: gemini-1.5-flash)",
    )
    parser.add_argument(
        "--tailor-extract-api-key-env",
        default=None,
        help="Env var name for extraction API key",
    )
    parser.add_argument(
        "--tailor-generate-provider",
        choices=["openrouter", "siliconflow", "groq", "custom"],
        default="custom",
        help="Generation provider for Step 2 (default: 'custom' — set explicitly to run tailoring)",
    )
    parser.add_argument(
        "--tailor-generate-model",
        default=None,
        help="Generation model for Step 2 (OpenRouter default: deepseek/deepseek-chat:free)",
    )
    parser.add_argument(
        "--tailor-generate-base-url",
        default=None,
        help="Override generation API base URL",
    )
    parser.add_argument(
        "--tailor-generate-api-key-env",
        default=None,
        help="Env var name for generation API key",
    )
    parser.add_argument(
        "--tailor-temperature",
        type=float,
        default=0.2,
        help="Generation temperature for tailoring",
    )
    parser.add_argument(
        "--tailor-retries",
        type=int,
        default=3,
        help="Retry attempts for API calls in tailoring step",
    )
    parser.add_argument(
        "--tailor-use-summarization",
        action="store_true",
        help="Summarize job descriptions before tailoring extraction",
    )
    parser.add_argument(
        "--tailor-compile-pdf",
        action="store_true",
        help="Compile generated .tex files to PDF using pdflatex",
    )
    parser.add_argument(
        "--tailor-latex-engine",
        default="pdflatex",
        help="LaTeX engine command for PDF compilation",
    )
    parser.add_argument(
        "--tailor-latex-timeout-sec",
        type=int,
        default=60,
        help="Timeout in seconds per LaTeX compile",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_matcher_subprocess(resume_text_file: str, top_n: int, use_embeddings: bool) -> None:
    project_root = _project_root()
    matcher_path = project_root / "src" / "matcher.py"

    command = [
        sys.executable,
        str(matcher_path),
        "--top-n",
        str(top_n),
    ]

    if resume_text_file:
        command.extend(["--resume-text-file", resume_text_file])
    if use_embeddings:
        command.append("--use-embeddings")

    completed = subprocess.run(command, cwd=str(project_root), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Matcher failed with exit code {completed.returncode}")


def _run_tailor_subprocess(
    resume_tex_file: str,
    max_jobs: int,
    min_match_score: int,
    output_dir: str,
    extract_provider: str,
    extract_model: str | None,
    extract_api_key_env: str | None,
    generate_provider: str,
    generate_model: str | None,
    generate_base_url: str | None,
    generate_api_key_env: str | None,
    temperature: float,
    retries: int,
    use_summarization: bool,
    compile_pdf: bool,
    latex_engine: str,
    latex_timeout_sec: int,
) -> None:
    project_root = _project_root()
    tailor_path = project_root / "src" / "resume_tailor.py"

    command = [
        sys.executable,
        str(tailor_path),
        "--resume-tex-file",
        resume_tex_file,
        "--max-jobs",
        str(max_jobs),
        "--min-match-score",
        str(min_match_score),
        "--output-dir",
        output_dir,
        "--extract-provider",
        extract_provider,
        "--generate-provider",
        generate_provider,
        "--temperature",
        str(temperature),
        "--retries",
        str(retries),
        "--latex-engine",
        latex_engine,
        "--latex-timeout-sec",
        str(latex_timeout_sec),
    ]

    if extract_model:
        command.extend(["--extract-model", extract_model])
    if extract_api_key_env:
        command.extend(["--extract-api-key-env", extract_api_key_env])
    if generate_model:
        command.extend(["--generate-model", generate_model])
    if generate_base_url:
        command.extend(["--generate-base-url", generate_base_url])
    if generate_api_key_env:
        command.extend(["--generate-api-key-env", generate_api_key_env])
    if use_summarization:
        command.append("--use-summarization")
    if compile_pdf:
        command.append("--compile-pdf")

    completed = subprocess.run(command, cwd=str(project_root), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Resume tailoring failed with exit code {completed.returncode}")


def run_pipeline() -> None:
    args = parse_args()
    total_steps = 3 if args.run_tailor else 2

    if not args.skip_scrape:
        print(f"[INFO] Step 1/{total_steps}: Running scraper")
        asyncio.run(run_scraper())
    else:
        print(f"[INFO] Step 1/{total_steps}: Skipped scraper")

    print(f"[INFO] Step 2/{total_steps}: Running matcher")
    _run_matcher_subprocess(args.resume_text_file, args.top_n, args.use_embeddings)

    if args.run_tailor:
        if not args.resume_tex_file:
            raise ValueError("--resume-tex-file is required when using --run-tailor")

        print(f"[INFO] Step 3/{total_steps}: Running resume tailoring")
        _run_tailor_subprocess(
            resume_tex_file=args.resume_tex_file,
            max_jobs=args.tailor_max_jobs,
            min_match_score=args.tailor_min_match_score,
            output_dir=args.tailor_output_dir,
            extract_provider=args.tailor_extract_provider,
            extract_model=args.tailor_extract_model,
            extract_api_key_env=args.tailor_extract_api_key_env,
            generate_provider=args.tailor_generate_provider,
            generate_model=args.tailor_generate_model,
            generate_base_url=args.tailor_generate_base_url,
            generate_api_key_env=args.tailor_generate_api_key_env,
            temperature=args.tailor_temperature,
            retries=args.tailor_retries,
            use_summarization=args.tailor_use_summarization,
            compile_pdf=args.tailor_compile_pdf,
            latex_engine=args.tailor_latex_engine,
            latex_timeout_sec=args.tailor_latex_timeout_sec,
        )

    print("[INFO] Pipeline completed")


if __name__ == "__main__":
    run_pipeline()
