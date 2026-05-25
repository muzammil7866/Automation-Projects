
# Instagram Follow Bot

## Overview

This repository is a Selenium automation template that collects users who liked a post and can optionally follow them. It is intentionally disabled by default for follow actions and now includes safer config validation, pinned dependencies, and repo hygiene files.

## Before You Run

This script is meant for workflows you are allowed to automate. Keep `ENABLE_FOLLOWING=false` unless you explicitly want follow actions enabled.

## Files

- `instagram_follow_bot.py` - login, likers collection, and follow workflow
- `requirements.txt` - runtime dependencies
- `.env.example` - local environment template
- `.gitignore` - excludes local, build, and cache files

## Setup

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` if you want to manage values locally.

Set these environment variables before running:

- `INSTAGRAM_USERNAME`
- `INSTAGRAM_PASSWORD`
- `INSTAGRAM_POST_URL`
- `ENABLE_FOLLOWING`
- `HEADLESS`
- `FOLLOW_DELAY`
- `SCROLL_PAUSE_TIME`
- `MAX_FOLLOWS`

## Run

```bash
python instagram_follow_bot.py
```

## Notes

Following is disabled by default. Use it only in contexts where you have permission to automate the workflow.

## Safety Defaults

- Follow actions are opt-in.
- Invalid numeric settings fail fast with a clear error.
- Browser caches and local environment files are ignored by git.
