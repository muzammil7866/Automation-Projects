"""Robust matcher for filtering and scoring scraped jobs.

Behavior:
- Load data/jobs.json
- Build resume profile from provided resume text
- Apply hard filters (for example seniority exclusion)
- Compute section-aware keyword relevance, resume overlap, and skill match/gap
- Save ranked output with transparent sub-scores in data/filtered_jobs.json

Optional embedding-based semantic matching:
- Use --use-embeddings to enable multiple API backends with fallback
- Provides embedding similarity score as primary metric

Run:
    python src/matcher.py
    python src/matcher.py --resume-text-file data/resume_match.txt --top-n 15
    python src/matcher.py --resume-text-file data/resume_match.txt --use-embeddings --top-n 15
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv

try:
    from llm_matcher import compute_resume_embedding, compute_similarity_score
    HAS_LLM_MATCHER = True
except ImportError:
    HAS_LLM_MATCHER = False


DEFAULT_KEYWORDS = ["AI", "ML", "LLM", "Python", "Deep Learning"]

STOP_WORDS: Set[str] = {
    "and",
    "the",
    "with",
    "for",
    "from",
    "that",
    "this",
    "your",
    "using",
    "into",
    "over",
    "under",
    "will",
    "have",
    "has",
    "are",
    "was",
    "were",
    "you",
    "our",
    "their",
    "them",
    "about",
    "across",
    "than",
    "then",
    "also",
    "only",
    "job",
    "remote",
}

EXCLUDED_TITLE_WORDS = {"senior", "sr", "sr.", "staff", "principal", "lead", "architect", "director"}

RESUME_CANDIDATE_KEYWORDS = [
    "AI",
    "ML",
    "LLM",
    "Python",
    "Deep Learning",
    "Machine Learning",
    "NLP",
    "Computer Vision",
    "Data Science",
    "MLOps",
    "LangChain",
    "PyTorch",
    "TensorFlow",
    "Scikit-learn",
    "Generative AI",
    "Prompt Engineering",
    "RAG",
    "SQL",
    "FastAPI",
    "Docker",
    "AWS",
]

SKILL_ALIASES: Dict[str, List[str]] = {
    "python": ["python"],
    "sql": ["sql", "postgres", "mysql"],
    "ai": ["ai", "artificial intelligence"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "llm": ["llm", "large language model", "large language models"],
    "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "langchain": ["langchain"],
    "pytorch": ["pytorch"],
    "tensorflow": ["tensorflow"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "fastapi": ["fastapi"],
    "aws": ["aws", "amazon web services"],
    "docker": ["docker"],
    "mlops": ["mlops", "ml ops"],
    "airflow": ["airflow", "apache airflow"],
    "mlflow": ["mlflow"],
    "openai": ["openai", "openai api"],
    "transformers": ["transformers", "hugging face"],
}


@dataclass(frozen=True)
class MatcherConfig:
    jobs_path: str = "data/jobs.json"
    output_path: str = "data/filtered_jobs.json"
    top_n: int = 20
    exclude_senior_titles: bool = True


def _resolve_project_path(relative_path: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return (project_root / relative_path).resolve()


def _normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def _contains_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    if " " in phrase:
        return phrase in text
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


def _count_occurrences(text: str, phrase: str) -> int:
    if not text or not phrase:
        return 0
    if " " in phrase:
        return len(re.findall(re.escape(phrase), text, flags=re.IGNORECASE))
    return len(re.findall(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE))


def _job_text(job: Dict[str, str]) -> str:
    return " ".join(
        [
            job.get("job_title", ""),
            job.get("job_description", ""),
            job.get("listing_snippet", ""),
            job.get("extracted_job_description", ""),
            job.get("company", ""),
            job.get("location", ""),
        ]
    )


def _split_description_sections(description: str) -> Dict[str, str]:
    """Best-effort parsing of common JD sections."""
    text = description or ""
    lowered = text.lower()

    markers = {
        "requirements": ["requirements", "required", "what we're looking for", "qualifications"],
        "responsibilities": ["responsibilities", "what you'll do", "what you will do", "how will you make an impact"],
        "nice_to_have": ["nice to have", "preferred", "bonus"],
    }

    section_spans: List[Tuple[int, str]] = []
    for section, words in markers.items():
        for word in words:
            idx = lowered.find(word)
            if idx >= 0:
                section_spans.append((idx, section))
                break

    section_spans.sort(key=lambda x: x[0])
    if not section_spans:
        return {"requirements": "", "responsibilities": text, "nice_to_have": ""}

    sections = {"requirements": "", "responsibilities": "", "nice_to_have": ""}
    for i, (start, section) in enumerate(section_spans):
        end = section_spans[i + 1][0] if i + 1 < len(section_spans) else len(text)
        chunk = _normalize_space(text[start:end])
        if len(chunk) > len(sections.get(section, "")):
            sections[section] = chunk

    return sections


def load_jobs(jobs_path: str) -> List[Dict[str, str]]:
    path = _resolve_project_path(jobs_path)
    if not path.exists():
        raise FileNotFoundError(f"jobs file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("jobs.json must contain a list of job dictionaries")

    normalized_jobs: List[Dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        normalized_jobs.append(
            {
                "job_title": str(item.get("job_title", "")),
                "company": str(item.get("company", "")),
                "location": str(item.get("location", "")),
                "job_description": str(item.get("job_description", "")),
                "listing_snippet": str(item.get("listing_snippet", "")),
                "extracted_job_description": str(item.get("extracted_job_description", "")),
                "description_source": str(item.get("description_source", "")),
                "apply_link": str(item.get("apply_link", "")),
            }
        )
    return normalized_jobs


def _read_resume_text(resume_text_file: str) -> str:
    path = Path(resume_text_file)
    if not path.is_absolute():
        path = _resolve_project_path(resume_text_file)

    if not path.exists():
        raise FileNotFoundError(f"resume text file not found: {path}")

    if path.suffix.lower() not in {".txt", ".tex", ".md"}:
        raise ValueError("resume text file should be .txt, .tex, or .md for this matcher")

    return path.read_text(encoding="utf-8", errors="ignore")


def infer_keywords_from_resume(resume_text: str) -> List[str]:
    lowered = resume_text.lower()
    found = [kw for kw in RESUME_CANDIDATE_KEYWORDS if kw.lower() in lowered]
    deduped: List[str] = []
    for kw in found:
        if kw not in deduped:
            deduped.append(kw)
    return deduped[:14] if deduped else DEFAULT_KEYWORDS


def extract_resume_terms(resume_text: str, inferred_keywords: List[str]) -> List[str]:
    lowered = resume_text.lower()
    terms: List[str] = [kw.lower() for kw in inferred_keywords]

    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-\+\.]{2,}", lowered)
    freq: Dict[str, int] = {}
    for tok in tokens:
        if tok in STOP_WORDS or tok.isdigit():
            continue
        freq[tok] = freq.get(tok, 0) + 1

    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    for tok, _ in ranked[:50]:
        terms.append(tok)

    seen = set()
    unique_terms: List[str] = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        unique_terms.append(t)
    return unique_terms[:80]


def extract_profile_skills(text: str) -> Set[str]:
    lowered = text.lower()
    skills: Set[str] = set()
    for canonical, aliases in SKILL_ALIASES.items():
        if any(_contains_phrase(lowered, alias) for alias in aliases):
            skills.add(canonical)
    return skills


def _has_excluded_title_word(title: str) -> bool:
    lowered = title.lower()
    return any(_contains_phrase(lowered, w) for w in EXCLUDED_TITLE_WORDS)


def _section_keyword_score(job: Dict[str, str], keywords: List[str]) -> int:
    """Score 0-100 with section-aware weighting and diminishing returns."""
    title = (job.get("job_title", "") or "").lower()
    desc = (job.get("job_description", "") or "").lower()
    sections = _split_description_sections(desc)

    title_hits = 0
    req_hits = 0
    resp_hits = 0
    other_hits = 0

    for kw in keywords:
        k = kw.lower()
        title_hits += _count_occurrences(title, k)
        req_hits += _count_occurrences(sections.get("requirements", ""), k)
        resp_hits += _count_occurrences(sections.get("responsibilities", ""), k)
        all_hits = _count_occurrences(desc, k)
        other_hits += max(0, all_hits - req_hits - resp_hits)

    weighted = (4.0 * title_hits) + (3.0 * req_hits) + (2.2 * resp_hits) + (1.2 * other_hits)
    damped = math.log1p(weighted)
    max_ref = math.log1p(max(1.0, 6.0 * len(keywords)))
    score = round((damped / max_ref) * 100)
    return max(0, min(100, score))


def _resume_overlap_score(job: Dict[str, str], resume_terms: List[str]) -> int:
    if not resume_terms:
        return 0

    text = _job_text(job).lower()
    matches = 0
    for term in resume_terms:
        t = term.lower()
        if _contains_phrase(text, t):
            matches += 1

    ratio = matches / max(1, len(resume_terms))
    return max(0, min(100, round(ratio * 100)))


def _skill_match_and_gaps(job: Dict[str, str], resume_skills: Set[str]) -> Tuple[int, List[str], List[str]]:
    text = _job_text(job)
    job_skills = extract_profile_skills(text)

    if not job_skills:
        return 0, [], []

    matched = sorted(list(job_skills & resume_skills))
    missing = sorted(list(job_skills - resume_skills))
    score = round((len(matched) / max(1, len(job_skills))) * 100)
    return max(0, min(100, score)), matched[:20], missing[:20]


def _passes_keyword_gate(job: Dict[str, str], keywords: List[str]) -> bool:
    searchable = _job_text(job).lower()
    return any(_count_occurrences(searchable, kw.lower()) > 0 for kw in keywords)


def filter_and_score_jobs(
    jobs: List[Dict[str, str]],
    keywords: List[str],
    resume_terms: List[str],
    resume_skills: Set[str],
    exclude_senior_titles: bool,
    use_embeddings: bool = False,
    resume_embedding = None,
    gemini_key: str | None = None,
    hf_key: str | None = None,
) -> List[Dict[str, str]]:
    filtered: List[Dict[str, str]] = []

    for job in jobs:
        title = job.get("job_title", "")
        if exclude_senior_titles and _has_excluded_title_word(title):
            continue
        if not _passes_keyword_gate(job, keywords):
            continue

        # Compute embedding-based similarity if enabled
        embedding_score = 0
        embedding_source = ""
        if use_embeddings and resume_embedding is not None:
            try:
                job_description = (job.get("job_description", "") or "") + " " + (job.get("listing_snippet", "") or "")
                embedding_score, embedding_source = compute_similarity_score(
                    resume_embedding,
                    job_description,
                    gemini_key,
                    hf_key,
                )
            except Exception as exc:
                logging.warning("Embedding similarity failed for job '%s': %s", title, exc)
                embedding_score = 0

        section_keyword_score = _section_keyword_score(job, keywords)
        resume_overlap_score = _resume_overlap_score(job, resume_terms)
        skill_match_score, matched_skills, missing_skills = _skill_match_and_gaps(job, resume_skills)

        # Use embedding score as primary if available, otherwise blend text-based scores
        if use_embeddings and embedding_score > 0:
            match_score = embedding_score
        else:
            match_score = round(
                (0.45 * section_keyword_score)
                + (0.30 * skill_match_score)
                + (0.25 * resume_overlap_score)
            )
        match_score = max(0, min(100, match_score))

        scored_job = dict(job)
        scored_job["embedding_score"] = embedding_score
        scored_job["embedding_source"] = embedding_source
        scored_job["keyword_score"] = section_keyword_score
        scored_job["resume_overlap_score"] = resume_overlap_score
        scored_job["skill_match_score"] = skill_match_score
        scored_job["matched_skills"] = matched_skills
        scored_job["missing_skills"] = missing_skills
        scored_job["match_score"] = match_score
        filtered.append(scored_job)

    filtered.sort(key=lambda item: item.get("match_score", 0), reverse=True)
    return filtered


def save_filtered_jobs(filtered_jobs: List[Dict[str, str]], output_path: str, top_n: int) -> Path:
    path = _resolve_project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top_jobs = filtered_jobs[: max(0, top_n)]
    with path.open("w", encoding="utf-8") as f:
        json.dump(top_jobs, f, indent=2, ensure_ascii=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust filter and scorer for AI/ML job relevance")
    parser.add_argument("--jobs-path", default="data/jobs.json", help="Path to input jobs.json")
    parser.add_argument("--output-path", default="data/filtered_jobs.json", help="Path for filtered output JSON")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top jobs to save")
    parser.add_argument(
        "--resume-text-file",
        default="",
        help="Optional path to resume text (.txt/.tex/.md). Recommended for robust matching.",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Optional explicit keywords list. Overrides default and resume inference.",
    )
    parser.add_argument(
        "--allow-senior-titles",
        action="store_true",
        help="Allow senior/staff/principal/lead/director/architect roles in matcher output.",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Use semantic embedding-based similarity scoring (requires API keys)",
    )
    parser.add_argument(
        "--gemini-api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable for Gemini API key",
    )
    parser.add_argument(
        "--hf-api-key-env",
        default="HF_API_KEY",
        help="Environment variable for Hugging Face API key",
    )
    return parser.parse_args()


def run_matcher() -> List[Dict[str, str]]:
    args = parse_args()
    config = MatcherConfig(
        jobs_path=args.jobs_path,
        output_path=args.output_path,
        top_n=args.top_n,
        exclude_senior_titles=not args.allow_senior_titles,
    )

    resume_text = ""
    if args.resume_text_file:
        resume_text = _read_resume_text(args.resume_text_file)

    if args.keywords:
        keywords = args.keywords
    elif resume_text:
        keywords = infer_keywords_from_resume(resume_text)
    else:
        keywords = DEFAULT_KEYWORDS

    resume_terms = extract_resume_terms(resume_text, keywords) if resume_text else [k.lower() for k in keywords]
    resume_skills = extract_profile_skills(resume_text) if resume_text else set(k.lower() for k in keywords)

    # Setup embedding if requested
    resume_embedding = None
    gemini_key = None
    hf_key = None
    
    if args.use_embeddings:
        if not HAS_LLM_MATCHER:
            print("[WARNING] --use-embeddings requested but llm_matcher module unavailable. Falling back to text-based matching.")
            args.use_embeddings = False
        elif not resume_text:
            print("[WARNING] --use-embeddings requires --resume-text-file. Falling back to text-based matching.")
            args.use_embeddings = False
        else:
            load_dotenv()
            gemini_key = os.getenv(args.gemini_api_key_env)
            hf_key = os.getenv(args.hf_api_key_env)
            
            if not gemini_key and not hf_key:
                print(f"[WARNING] No API keys found ({args.gemini_api_key_env}, {args.hf_api_key_env}). Falling back to text-based matching.")
                args.use_embeddings = False
            else:
                try:
                    print("[INFO] Computing resume embedding for semantic matching...")
                    resume_embedding = compute_resume_embedding(resume_text, gemini_key, hf_key)
                    print("[INFO] Resume embedding computed successfully")
                except Exception as exc:
                    print(f"[WARNING] Embedding computation failed: {exc}. Falling back to text-based matching.")
                    args.use_embeddings = False
                    resume_embedding = None

    jobs = load_jobs(config.jobs_path)
    filtered_jobs = filter_and_score_jobs(
        jobs=jobs,
        keywords=keywords,
        resume_terms=resume_terms,
        resume_skills=resume_skills,
        exclude_senior_titles=config.exclude_senior_titles,
        use_embeddings=args.use_embeddings,
        resume_embedding=resume_embedding,
        gemini_key=gemini_key,
        hf_key=hf_key,
    )
    output_file = save_filtered_jobs(filtered_jobs, config.output_path, config.top_n)

    print(f"[INFO] Loaded {len(jobs)} jobs")
    print(f"[INFO] Using keywords: {keywords}")
    print(f"[INFO] Resume terms: {len(resume_terms)} | Resume skills: {len(resume_skills)}")
    print(f"[INFO] Matched {len(filtered_jobs)} jobs")
    print(f"[INFO] Matching mode: {'embedding-based' if args.use_embeddings else 'text-based'}")
    print(f"[INFO] Saved top {min(config.top_n, len(filtered_jobs))} jobs to: {output_file}")

    return filtered_jobs


if __name__ == "__main__":
    run_matcher()
