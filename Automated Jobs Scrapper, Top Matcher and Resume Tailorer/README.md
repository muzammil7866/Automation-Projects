# Automated Jobs Scrapper, Top Matcher and Resume Tailorer — MVP

This repository is a small, portfolio-ready MVP that automates the early stages of job hunting:

- Scrape roles
- Rank jobs against a local resume
- Optionally generate tailored LaTeX resumes for top matches

The project is intentionally scoped as an MVP. The `apply` step (submitting applications) is a future extension.

**Key scripts**

- `src/scraper.py` — scrape job listings
- `src/matcher.py` — rank jobs against a resume
- `src/resume_tailor.py` — optional two-step LLM tailoring flow
- `src/main.py` — pipeline orchestration

**Repo layout**

See the repository root for `src/`, `config/`, `data/`, and `output/` folders.

## Quick start (MVP)

1. Create and activate a Python virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
```

2. Configure keys (optional): copy `.env.example` to `.env` and fill provider API keys only if you plan to run tailoring or summarization.

3. Provide local resume inputs (these are gitignored): place `resume_match.txt` and/or `my_resume.tex` in the `data/` folder.

## Run examples

Run the full scrape + match pipeline:

```powershell
python src/main.py --top-n 15 --resume-text-file data/resume_match.txt
```

Run with tailoring (only if you have a `.tex` resume and API keys configured):

```powershell
python src/main.py --top-n 15 --resume-text-file data/resume_match.txt --run-tailor --resume-tex-file data/my_resume.tex --tailor-max-jobs 3
```

## Notes for publishing

- `data/`, `output/`, and `job-automation/browser_profile/` are excluded via `.gitignore` to avoid leaking personal files and large generated outputs.
- `src/apply.py` is a placeholder and intentionally left for future work.

## Next small improvements (recommended)

- Add a small, sanitized sample input in `data/sample/` to demonstrate the pipeline without secrets.
- Add unit tests for `matcher` scoring behavior.
- Document provider-specific env vars in `.env.example`.

## How the pipeline works (behind the scenes)

1. Scraping (`src/scraper.py`)
	 - Uses Playwright (headless Chromium) to navigate target job sites, collect job postings, and normalize fields into a JSON schema stored in `data/jobs.json`.
	 - Scraper logic includes simple deduplication and basic HTML parsing to extract title, company, location, and description.

2. Matching (`src/matcher.py` / `src/llm_matcher.py`)
	 - `matcher.py` loads the scraped jobs and a local resume text file, computes a match score for each job, and writes the top-N matches to `data/filtered_jobs.json`.
	 - Matching uses a mix of keyword heuristics (skills/years/key phrases) and an optional embedding-based similarity path in `src/llm_matcher.py`.
	 - When `--use-embeddings` is enabled, the code attempts to call available embedding providers (Gemini or Hugging Face) using the environment variables `GEMINI_API_KEY` and/or `HF_API_KEY` (see security section).

3. (Optional) Summarization (`src/llm_summarizer.py`)
	 - When enabled, job descriptions can be summarized before tailoring to reduce token consumption and focus the extraction step.

4. Resume tailoring (`src/resume_tailor.py`)
	 - Two-step LLM flow: (A) extract role-relevant sections from the source resume, (B) generate a tailored LaTeX resume (.tex) for each selected job.
	 - Tailoring supports multiple providers via CLI flags (OpenRouter, Groq, SiliconFlow, or `custom`). The provider choice determines which env var and API layout the code looks for.
	 - Output files (generated .tex and optional PDF) are written under `output/tailored_resumes/`.

5. Orchestration (`src/main.py`)
	 - `src/main.py` coordinates the pipeline and invokes `matcher.py` and `resume_tailor.py` as subprocesses so each step can run in isolation.
