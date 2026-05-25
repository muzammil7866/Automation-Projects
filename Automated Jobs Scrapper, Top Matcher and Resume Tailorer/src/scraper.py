"""Job scraper for AI/ML roles using Playwright for real browser automation.

MVP behavior:
- Scrape Indeed listing pages using Playwright (real browser, no bot detection)
- Filter results for AI/ML niche roles
- Export normalized job dictionaries to data/jobs.json

Run:
    python src/scraper.py
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
from urllib.parse import parse_qs, urlencode, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


@dataclass(frozen=True)
class ScraperConfig:
    """Configuration values for scraping and output."""

    headless: bool = True
    timeout_ms: int = 30000
    pause_between_requests_sec: float = 1.2
    pause_jitter_sec: float = 0.8
    max_jobs: int = 40
    max_listing_urls: int = 1
    max_items_per_url: int = 2
    location: str = "Remote"
    posted_within_days: int = 3
    sort_by_date: bool = True
    full_time_only: bool = True
    include_junior_roles: bool = True
    include_internship_roles: bool = True
    strict_title_filter: bool = True
    exclude_seniority_levels: tuple = ("senior", "sr", "sr.", "staff", "principal", "lead", "architect")
    fetch_detailed_descriptions: bool = True
    max_description_fetches_per_url: int = 3
    max_description_chars: int = 25000
    output_path: str = "data/jobs.json"


class JobScraper:
    """Scrapes Indeed job listings using Playwright and exports AI/ML jobs to JSON."""

    INDEED_BASE = "https://www.indeed.com"
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    BASE_ROLE_QUERIES = (
        "ai engineer",
        "agentic ai developer",
        "mlops engineer",
        "generative ai engineer",
        "computer vision engineer",
        "nlp engineer",
        "ml engineer",
        "machine learning engineer",
        "ai automation engineer",
        "python developer",
        "data scientist",
    )

    JUNIOR_ROLE_QUERIES = (
        "junior machine learning engineer",
        "junior ml engineer",
        "junior ai engineer",
        "entry level data scientist",
    )

    INTERNSHIP_ROLE_QUERIES = (
        "machine learning intern",
        "ai intern",
        "data science intern",
    )

    AI_ML_KEYWORDS = {
        "ai",
        "artificial intelligence",
        "ml",
        "machine learning",
        "deep learning",
        "llm",
        "nlp",
        "computer vision",
        "data scientist",
        "prompt engineer",
        "generative ai",
    }

    def __init__(self, config: Optional[ScraperConfig] = None) -> None:
        self.config = config or ScraperConfig()
        self._browser = None

    def _build_listing_urls(self) -> List[str]:
        """Build Indeed listing URLs from configured query filters."""
        core_roles = list(self.BASE_ROLE_QUERIES[:6])
        extra_roles = list(self.BASE_ROLE_QUERIES[6:])

        role_queries: List[str] = core_roles
        if self.config.include_junior_roles:
            role_queries.extend(self.JUNIOR_ROLE_QUERIES[:2])
        if self.config.include_internship_roles:
            role_queries.extend(self.INTERNSHIP_ROLE_QUERIES[:1])
        role_queries.extend(extra_roles)

        if self.config.include_junior_roles:
            role_queries.extend(self.JUNIOR_ROLE_QUERIES[2:])
        if self.config.include_internship_roles:
            role_queries.extend(self.INTERNSHIP_ROLE_QUERIES[1:])

        urls: List[str] = []
        for query in role_queries:
            params = {
                "q": query,
                "l": self.config.location,
            }

            if self.config.full_time_only:
                params["jt"] = "fulltime"
            if self.config.posted_within_days > 0:
                params["fromage"] = str(self.config.posted_within_days)
            if self.config.sort_by_date:
                params["sort"] = "date"

            urls.append(f"{self.INDEED_BASE}/jobs?{urlencode(params)}")

        return urls[: self.config.max_listing_urls]

    async def _sleep_with_jitter(self) -> None:
        delay = self.config.pause_between_requests_sec + random.uniform(0, self.config.pause_jitter_sec)
        await asyncio.sleep(delay)


    async def scrape(self) -> List[Dict[str, str]]:
        """Scrape all configured sources and return AI/ML jobs only."""
        jobs: List[Dict[str, str]] = []
        seen_keys: Set[str] = set()
        listing_urls = self._build_listing_urls()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.config.headless)
            self._browser = browser
            context = await browser.new_context(
                user_agent=self.USER_AGENT
            )

            for index, url in enumerate(listing_urls, start=1):
                print(f"[INFO] URL {index}/{len(listing_urls)}")
                listing_jobs = await self._scrape_indeed_listing(context, url)

                for job in listing_jobs:
                    dedupe_key = self._dedupe_key(job)
                    if dedupe_key in seen_keys:
                        continue
                    seen_keys.add(dedupe_key)
                    jobs.append(job)

                    if len(jobs) >= self.config.max_jobs:
                        await context.close()
                        await browser.close()
                        return jobs

                    await self._sleep_with_jitter()

            await context.close()
            await browser.close()
            self._browser = None

        return jobs

    async def _scrape_indeed_listing(self, context, listing_url: str) -> List[Dict[str, str]]:
        """Scrape one Indeed listing page using Playwright and extract AI/ML jobs."""
        page = await context.new_page()
        extracted_jobs: List[Dict[str, str]] = []
        fetched_descriptions = 0

        try:
            print(f"[INFO] Loading {listing_url}...")
            await page.goto(listing_url, wait_until="load", timeout=self.config.timeout_ms)
            
            # Wait for job cards to load
            try:
                await page.wait_for_selector("div.job_seen_beacon, div.cardOutline", timeout=5000)
            except:
                print(f"[WARN] Timeout waiting for job cards, proceeding with available content")
            
            # Get page HTML after JavaScript rendering
            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")
            
            cards = soup.select("div.job_seen_beacon") or soup.select("div.cardOutline")
            print(f"[INFO] Found {len(cards)} job cards on page")

            for i, card in enumerate(cards):
                title = self._clean_text(
                    self._first_text(
                        card,
                        [
                            "h2.jobTitle",
                            "h2.jobTitle span[title]",
                            "a[data-jk] span",
                        ],
                    )
                )

                # Try multiple selectors for company name from the card itself
                company = self._clean_text(
                    self._first_text(
                        card,
                        [
                            "span.companyName",
                            "div.company_location span.companyName",
                            "[data-testid='company-name']",
                            "a[data-testid='employer-name']",
                            ".company-name",
                        ],
                    )
                )

                location = self._clean_text(
                    self._first_text(
                        card,
                        [
                            "div.company_location div.companyLocation",
                            "div.companyLocation",
                            "span.companyLocation",
                            "[data-testid='job-search-card-location']",
                        ],
                    )
                )

                job_link = self._first_attr(
                    card,
                    [
                        "h2.jobTitle a",
                        "a.jcs-JobTitle",
                        "a[data-jk]",
                    ],
                    "href",
                )
                apply_link = self._to_absolute_url(job_link)

                # Get snippet from the card listing
                snippet = self._clean_text(
                    self._first_text(
                        card,
                        [
                            "div.job-snippet",
                            "ul.job-snippet",
                            "[data-testid='job-snippet']",
                        ],
                    )
                )
                if not snippet:
                    # Fallback: capture whatever visible text exists inside the card.
                    snippet = self._clean_text(card.get_text(" ", strip=True))

                description = snippet if snippet else "Job listing - visit Indeed for full details"
                extracted_detail = ""
                description_source = "listing_snippet"

                # Filter first so description-fetch budget is spent only on relevant jobs.
                if not self._is_ai_ml_role(title=title, description=description, tags=[]):
                    continue

                if (
                    self.config.fetch_detailed_descriptions
                    and apply_link
                    and fetched_descriptions < self.config.max_description_fetches_per_url
                ):
                    detailed_description = await self._fetch_job_description(context, apply_link)
                    if not detailed_description:
                        detailed_description = await self._fetch_job_description(
                            context, self._to_viewjob_url(apply_link)
                        )
                    if detailed_description:
                        description = detailed_description
                        extracted_detail = detailed_description
                        description_source = "detail_page"
                        print(f"[INFO] Extracted full description ({len(detailed_description)} chars)")
                    else:
                        print("[WARN] Detail description unavailable, using listing snippet")
                    fetched_descriptions += 1

                job = {
                    "job_title": title or "N/A",
                    "company": company or "N/A",
                    "location": location or "Remote",
                    "job_description": description or "N/A",
                    "listing_snippet": snippet or "",
                    "extracted_job_description": extracted_detail,
                    "description_source": description_source,
                    "apply_link": apply_link or listing_url,
                }
                extracted_jobs.append(job)
                print(f"[INFO] ({i+1}) Added job: {title} @ {company}")

                if len(extracted_jobs) >= self.config.max_items_per_url:
                    break

        except Exception as exc:
            print(f"[WARN] Error scraping {listing_url}: {exc}")
        finally:
            await page.close()

        return extracted_jobs

    async def _fetch_job_description(self, context, job_url: str) -> str:
        """Fetch an Indeed detail page and return a cleaned description text."""
        temp_context = None
        fetch_context = context
        if self._browser is not None:
            temp_context = await self._browser.new_context(user_agent=self.USER_AGENT)
            fetch_context = temp_context

        page = await fetch_context.new_page()

        try:
            for attempt in range(2):
                if attempt > 0:
                    await asyncio.sleep(1.5)

                await page.goto(job_url, wait_until="load", timeout=self.config.timeout_ms)
                try:
                    await page.wait_for_selector(
                        "div#jobDescriptionText, div.jobsearch-jobDescriptionText, div[data-testid='jobsearch-JobComponent-description']",
                        timeout=8000,
                    )
                except Exception:
                    # Keep going with whatever content is currently rendered.
                    pass

                html = await page.content()
                soup = BeautifulSoup(html, "html.parser")

                description_node = (
                    soup.select_one("div#jobDescriptionText")
                    or soup.select_one("div.jobsearch-jobDescriptionText")
                    or soup.select_one("div[data-testid='jobsearch-JobComponent-description']")
                    or soup.select_one("main")
                )

                if description_node is None:
                    continue

                paragraphs = [self._clean_text(p.get_text(" ", strip=True)) for p in description_node.select("p, li")]
                paragraphs = [p for p in paragraphs if p]

                if paragraphs:
                    text = "\n".join(paragraphs)
                else:
                    text = self._clean_text(description_node.get_text(" ", strip=True))

                # Keep descriptions rich but bounded to avoid oversized JSON payloads.
                if len(text) > self.config.max_description_chars:
                    text = text[: self.config.max_description_chars].rstrip()

                lowered = text.lower()
                if "cloudflare" in lowered or "ray id" in lowered:
                    continue
                if len(text.strip()) < 80:
                    continue
                return text

            return ""

        except Exception as exc:
            print(f"[WARN] Error fetching description from {job_url}: {exc}")
            return ""
        finally:
            await page.close()
            if temp_context is not None:
                await temp_context.close()

    def _is_ai_ml_role(self, title: str, description: str, tags: List[str]) -> bool:
        """Determine whether a job belongs to AI/ML niche and is not excluded by seniority."""
        title_text = (title or "").lower()
        description_text = (description or "").lower()
        tag_text = " ".join(tags).lower()

        # Check if job title contains excluded seniority levels
        for excluded_word in self.config.exclude_seniority_levels:
            if excluded_word in title_text:
                return False

        title_hit = any(keyword in title_text for keyword in self.AI_ML_KEYWORDS)
        context_hit = any(keyword in f"{description_text} {tag_text}" for keyword in self.AI_ML_KEYWORDS)

        if self.config.strict_title_filter:
            return title_hit
        return title_hit or context_hit

    def save_jobs(self, jobs: Iterable[Dict[str, str]]) -> Path:
        """Save jobs list to JSON output path and return the file path."""
        output_path = Path(self.config.output_path)

        # Resolve relative path from project root when script runs from src/.
        if not output_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            output_path = project_root / output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_jobs = list(jobs)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(serialized_jobs, f, indent=2, ensure_ascii=False)

        return output_path


    @staticmethod
    def _first_text(parent: BeautifulSoup, selectors: List[str]) -> str:
        for selector in selectors:
            node = parent.select_one(selector)
            if node is not None:
                if node.has_attr("title"):
                    return str(node.get("title", "")).strip() or node.get_text(" ", strip=True)
                return node.get_text(" ", strip=True)
        return ""

    @staticmethod
    def _first_attr(parent: BeautifulSoup, selectors: List[str], attr: str) -> str:
        for selector in selectors:
            node = parent.select_one(selector)
            if node is not None and node.has_attr(attr):
                return str(node.get(attr, "")).strip()
        return ""

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join((text or "").split())

    @staticmethod
    def _dedupe_key(job: Dict[str, str]) -> str:
        return "|".join(
            [
                job.get("job_title", "").strip().lower(),
                job.get("company", "").strip().lower(),
                job.get("apply_link", "").strip().lower(),
            ]
        )

    @staticmethod
    def _to_absolute_url(maybe_relative: str) -> str:
        if not maybe_relative:
            return ""
        if maybe_relative.startswith("http"):
            return maybe_relative
        if maybe_relative.startswith("/"):
            return f"{JobScraper.INDEED_BASE}{maybe_relative}"
        return f"{JobScraper.INDEED_BASE}/{maybe_relative}"

    @staticmethod
    def _to_viewjob_url(job_url: str) -> str:
        """Convert click-tracking URLs to stable /viewjob URL when possible."""
        if not job_url:
            return ""

        parsed = urlparse(job_url)
        query = parse_qs(parsed.query)
        job_key = query.get("jk", [""])[0]
        if job_key:
            return f"{JobScraper.INDEED_BASE}/viewjob?jk={job_key}"
        return job_url

async def run_scraper() -> List[Dict[str, str]]:
    """Entrypoint used by main and direct script execution."""
    scraper = JobScraper()
    jobs = await scraper.scrape()
    output_file = scraper.save_jobs(jobs)

    print(f"[INFO] Scraped {len(jobs)} AI/ML jobs.")
    print(f"[INFO] Saved results to: {output_file}")
    return jobs


if __name__ == "__main__":
    asyncio.run(run_scraper())

