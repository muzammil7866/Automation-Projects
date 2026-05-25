"""Configurable Instagram automation template.

This keeps the original idea but removes hardcoded credentials and makes the
automation opt-in through environment variables.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


LOGGER = logging.getLogger("instagram_follow_bot")


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class BotConfig:
    username: str
    password: str
    post_url: str
    follow_delay: float = 5.0
    scroll_pause_time: float = 2.0
    max_follows: int = 40
    headless: bool = False
    enable_following: bool = False


def _read_bool(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _read_positive_float(name: str, default: str) -> float:
    try:
        value = float(os.getenv(name, default))
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc

    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than 0")
    return value


def _read_positive_int(name: str, default: str) -> int:
    try:
        value = int(os.getenv(name, default))
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc

    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than 0")
    return value


def load_config() -> BotConfig:
    username = os.getenv("INSTAGRAM_USERNAME", "").strip()
    password = os.getenv("INSTAGRAM_PASSWORD", "").strip()
    post_url = os.getenv("INSTAGRAM_POST_URL", "").strip()

    if not username:
        raise ConfigurationError("INSTAGRAM_USERNAME is required")
    if not password:
        raise ConfigurationError("INSTAGRAM_PASSWORD is required")
    if not post_url:
        raise ConfigurationError("INSTAGRAM_POST_URL is required")

    return BotConfig(
        username=username,
        password=password,
        post_url=post_url,
        follow_delay=_read_positive_float("FOLLOW_DELAY", "5"),
        scroll_pause_time=_read_positive_float("SCROLL_PAUSE_TIME", "2"),
        max_follows=_read_positive_int("MAX_FOLLOWS", "40"),
        headless=_read_bool("HEADLESS"),
        enable_following=_read_bool("ENABLE_FOLLOWING"),
    )


def create_driver(headless: bool = False) -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,900")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def wait_for_username_and_password(driver: webdriver.Chrome) -> Tuple[object, object]:
    wait = WebDriverWait(driver, 20)
    username_input = wait.until(EC.presence_of_element_located((By.NAME, "username")))
    password_input = wait.until(EC.presence_of_element_located((By.NAME, "password")))
    return username_input, password_input


def find_first_clickable(
    driver: webdriver.Chrome,
    locators: List[Tuple[str, str]],
    timeout: int = 10,
):
    wait = WebDriverWait(driver, timeout)
    last_exc: Optional[Exception] = None
    for locator in locators:
        try:
            return wait.until(EC.element_to_be_clickable(locator))
        except Exception as exc:  # pragma: no cover - browser/runtime dependency
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No locators were provided")


def login(driver: webdriver.Chrome, config: BotConfig) -> None:
    driver.get("https://www.instagram.com/accounts/login/")
    username_input, password_input = wait_for_username_and_password(driver)
    username_input.send_keys(config.username)
    password_input.send_keys(config.password)
    password_input.send_keys(Keys.RETURN)
    time.sleep(5)


def collect_likers(driver: webdriver.Chrome, post_url: str, scroll_pause_time: float) -> set[str]:
    driver.get(post_url)
    time.sleep(random.uniform(5, 10))

    try:
        likes_button = find_first_clickable(
            driver,
            [
                (By.XPATH, "//a[contains(@href, 'liked_by')]") ,
                (By.XPATH, "//button[contains(., 'likes') or contains(., 'other')]") ,
            ],
            timeout=15,
        )
        driver.execute_script("arguments[0].click();", likes_button)
        time.sleep(random.uniform(5, 8))
    except Exception as exc:  # pragma: no cover - browser/runtime dependency
        LOGGER.warning("Could not open the likers list: %s", exc)
        return set()

    likers: set[str] = set()
    try:
        scroll_box = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//div[@role='dialog']//div[contains(@style, 'overflow')]"))
        )

        stable_rounds = 0
        previous_count = 0
        while stable_rounds < 3:
            driver.execute_script("arguments[0].scrollTop += 300;", scroll_box)
            time.sleep(random.uniform(scroll_pause_time, scroll_pause_time + 2))

            users = driver.find_elements(By.XPATH, "//div[@role='dialog']//a[contains(@href, '/')]")
            for user in users:
                href = user.get_attribute("href") or ""
                parts = href.rstrip("/").split("/")
                if parts:
                    likers.add(parts[-1])

            stable_rounds = stable_rounds + 1 if len(likers) == previous_count else 0
            previous_count = len(likers)
    except Exception as exc:  # pragma: no cover - browser/runtime dependency
        LOGGER.warning("Failed while collecting likers: %s", exc)

    return likers


def follow_users(driver: webdriver.Chrome, users: Iterable[str], config: BotConfig) -> int:
    if not config.enable_following:
        LOGGER.info("Following is disabled. Set ENABLE_FOLLOWING=true to allow follow actions.")
        return 0

    follow_count = 0
    for username in users:
        driver.get(f"https://www.instagram.com/{username}/")
        time.sleep(random.uniform(5, 8))

        try:
            follow_button = find_first_clickable(
                driver,
                [
                    (By.XPATH, "//div[contains(@class, '_ap3a') and contains(text(), 'Follow')]") ,
                    (By.XPATH, "//button[normalize-space()='Follow']") ,
                ],
                timeout=10,
            )
            if "Follow" in follow_button.text:
                driver.execute_script("arguments[0].click();", follow_button)
                follow_count += 1
                time.sleep(config.follow_delay)
        except Exception as exc:  # pragma: no cover - browser/runtime dependency
            LOGGER.info("Skipped %s: %s", username, exc)

        if follow_count >= config.max_follows:
            break

    return follow_count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config()
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
    if not (config.username and config.password and config.post_url):
        raise SystemExit("Set INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD, and INSTAGRAM_POST_URL before running.")

    driver = create_driver(headless=config.headless)
    try:
        login(driver, config)
        likers = collect_likers(driver, config.post_url, config.scroll_pause_time)
        LOGGER.info("Collected %s likers", len(likers))
        followed = follow_users(driver, likers, config)
        LOGGER.info("Followed %s users", followed)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
