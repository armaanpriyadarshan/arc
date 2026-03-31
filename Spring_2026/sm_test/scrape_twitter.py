"""
Twitter Hackathon Scraper — find hackathon announcements without the Twitter API.

Three-tier approach (tried in order when method=auto):
  1. Playwright  — headless Chromium browses twitter.com/search, scrolls for
                   pagination, extracts tweet data from the DOM.
  2. Google      — queries Google for site:x.com results, parses snippets.
  3. Nitter      — scrapes server-rendered HTML from a live Nitter mirror.

Setup:
  pip install -r requirements.txt
  python -m playwright install chromium   # only needed for Playwright method

Usage examples:
  # Auto mode (tries Playwright -> Google -> Nitter)
  python scrape_twitter.py

  # Google-only, 5 results, save to file
  python scrape_twitter.py --method google -q "hackathon SF April 2026" -n 5 -o results.json

  # Playwright with extra scrolling
  python scrape_twitter.py --method playwright --scroll-count 25

CLI flags:
  -q  / --queries       Comma-separated search terms
  -n  / --max-results   Max tweets to collect (default 50)
  -o  / --output        JSON output file path
  --method              auto | playwright | google | nitter
  --scroll-count        Playwright scroll iterations (default 15)
  --delay               Seconds between requests (default 2.0)
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

# ── ANSI Colors ───────────────────────────────────────────────────────────────

GREEN = "\033[32m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_QUERIES = [
    "hackathon SF April 2026",
    "hackathon NYC April 2026",
    "hackathon AI April 2026",
]

MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds — exponential backoff multiplier

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

NITTER_MIRRORS = [
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.woodland.cafe",
    "https://nitter.1d4.us",
    "https://nitter.kavin.rocks",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _jitter(base_delay):
    """Return base_delay plus a random jitter between 0 and 50 % of base."""
    return base_delay + random.uniform(0, base_delay * 0.5)


def _normalize_url(url):
    """Normalize twitter.com / x.com URLs so deduplication works across domains."""
    if url:
        url = url.replace("https://x.com/", "https://twitter.com/")
        url = url.replace("http://x.com/", "https://twitter.com/")
        url = url.replace("http://twitter.com/", "https://twitter.com/")
    return url


def _deduplicate(results):
    """Remove duplicate tweets based on normalized URL."""
    seen = set()
    unique = []
    for tweet in results:
        key = _normalize_url(tweet.get("url", ""))
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        unique.append(tweet)
    return unique


def _make_session():
    """Create a requests.Session with a realistic User-Agent."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


def _retry_request(session, method, url, max_retries=MAX_RETRIES, delay=2.0, **kwargs):
    """
    HTTP request with exponential backoff on timeouts and 429 responses.
    Returns the Response object on success, or None after all retries fail.
    """
    kwargs.setdefault("timeout", 15)
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.request(method, url, **kwargs)
            if resp.status_code == 429:
                wait = BACKOFF_BASE ** attempt + random.uniform(0, 1)
                print(f"  {GRAY}Rate limited (429). Waiting {wait:.1f}s "
                      f"(attempt {attempt}/{max_retries})...{RESET}")
                time.sleep(wait)
                continue
            return resp
        except requests.exceptions.Timeout:
            wait = BACKOFF_BASE ** attempt + random.uniform(0, 1)
            print(f"  {GRAY}Timeout. Retrying in {wait:.1f}s "
                  f"(attempt {attempt}/{max_retries})...{RESET}")
            time.sleep(wait)
        except requests.exceptions.RequestException as exc:
            print(f"  {GRAY}Request error: {exc}{RESET}")
            return None
    print(f"  {GRAY}All {max_retries} retries exhausted for {url}{RESET}")
    return None


def _safe_int(text):
    """Parse a human-readable count like '1.2K' or '340' into an integer."""
    if not text:
        return 0
    text = text.strip().replace(",", "")
    multiplier = 1
    if text.upper().endswith("K"):
        multiplier = 1_000
        text = text[:-1]
    elif text.upper().endswith("M"):
        multiplier = 1_000_000
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except (ValueError, TypeError):
        return 0


def _build_output(results, queries, method):
    """Build the JSON-serializable output dict."""
    return {
        "metadata": {
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "search_terms": queries,
            "method": method,
            "total_results": len(results),
        },
        "results": results,
    }


def _print_results(results):
    """Pretty-print results to the console."""
    if not results:
        print(f"\n  {GRAY}No results found.{RESET}\n")
        return

    print(f"\n  {GREEN}{BOLD}Found {len(results)} tweet(s):{RESET}\n")
    for i, tweet in enumerate(results, 1):
        author = tweet.get("author") or tweet.get("handle") or "unknown"
        handle = tweet.get("handle", "")
        text = tweet.get("text", "")
        url = tweet.get("url", "")
        likes = tweet.get("likes", 0)
        retweets = tweet.get("retweets", 0)
        ts = tweet.get("timestamp", "")

        print(f"  {GREEN}{BOLD}{i}.{RESET} {BOLD}{author}{RESET}", end="")
        if handle and handle != author:
            print(f" {GRAY}@{handle}{RESET}", end="")
        print()

        display_text = text[:280] + ("..." if len(text) > 280 else "")
        for line in display_text.split("\n"):
            print(f"     {line}")

        meta_parts = []
        if likes:
            meta_parts.append(f"likes: {likes}")
        if retweets:
            meta_parts.append(f"RTs: {retweets}")
        if ts:
            meta_parts.append(ts)
        if meta_parts:
            print(f"     {GRAY}{' | '.join(meta_parts)}{RESET}")
        if url:
            print(f"     {GRAY}{url}{RESET}")
        print()


# ── Tier 1: Playwright ────────────────────────────────────────────────────────

def scrape_playwright(queries, max_results=50, scroll_count=15, delay=2.0):
    """
    Scrape Twitter search results using headless Chromium via Playwright.

    Anti-scraping measures:
      - Realistic viewport (1920x1080) and User-Agent
      - navigator.webdriver flag disabled
      - --disable-blink-features=AutomationControlled
      - Random jitter between scrolls and page loads
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(f"\n  {BOLD}Playwright is not installed.{RESET}")
        print("  Install it with:")
        print(f"    {GREEN}pip install playwright{RESET}")
        print(f"    {GREEN}python -m playwright install chromium{RESET}\n")
        return None

    results = []
    print(f"\n  {BOLD}[Playwright]{RESET} Starting headless Chromium...\n")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
            )
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=USER_AGENT,
            )
            # Disable the navigator.webdriver flag that reveals automation
            context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', "
                "{get: () => undefined})"
            )
            page = context.new_page()

            for query in queries:
                if len(results) >= max_results:
                    break

                search_url = (
                    f"https://twitter.com/search?q={quote_plus(query)}&f=live"
                )
                print(f"  Searching: {query}")
                try:
                    page.goto(search_url, wait_until="domcontentloaded",
                              timeout=30_000)
                except Exception as exc:
                    print(f"  {GRAY}Page load failed: {exc}{RESET}")
                    continue

                time.sleep(_jitter(delay))

                for _ in range(scroll_count):
                    if len(results) >= max_results:
                        break
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    time.sleep(_jitter(delay * 0.4))

                # Parse tweet articles from the DOM
                articles = page.query_selector_all("article[data-testid='tweet']")
                for article in articles:
                    if len(results) >= max_results:
                        break
                    tweet = _parse_playwright_article(article, query)
                    if tweet:
                        results.append(tweet)

                time.sleep(_jitter(delay))

            browser.close()

    except Exception as exc:
        print(f"  {GRAY}Playwright error: {exc}{RESET}")
        if "Executable doesn't exist" in str(exc) or "browserType.launch" in str(exc):
            print(f"\n  {BOLD}Chromium browser not installed.{RESET}")
            print("  Run:")
            print(f"    {GREEN}python -m playwright install chromium{RESET}\n")
            return None

    return _deduplicate(results) if results else None


def _parse_playwright_article(article, source_query):
    """Extract tweet data from a Playwright ElementHandle for a tweet article."""
    try:
        author = ""
        handle = ""
        user_links = article.query_selector_all("a[role='link']")
        for link in user_links:
            href = link.get_attribute("href") or ""
            if href.startswith("/") and "/status/" not in href:
                span = link.query_selector("span")
                if span:
                    span_text = span.inner_text().strip()
                    if span_text.startswith("@"):
                        handle = span_text.lstrip("@")
                    elif span_text and not author:
                        author = span_text
                if author or handle:
                    break

        text_el = article.query_selector("div[data-testid='tweetText']")
        text = text_el.inner_text().strip() if text_el else ""

        time_el = article.query_selector("time")
        timestamp = ""
        if time_el:
            timestamp = time_el.get_attribute("datetime") or time_el.inner_text()

        url = ""
        time_link = article.query_selector("a[href*='/status/']")
        if time_link:
            href = time_link.get_attribute("href") or ""
            if href.startswith("/"):
                url = f"https://twitter.com{href}"
            elif href.startswith("http"):
                url = href

        likes = 0
        retweets = 0
        like_el = article.query_selector(
            "button[data-testid='like'] span, "
            "button[data-testid='unlike'] span"
        )
        if like_el:
            likes = _safe_int(like_el.inner_text())

        rt_el = article.query_selector("button[data-testid='retweet'] span")
        if rt_el:
            retweets = _safe_int(rt_el.inner_text())

        if not text and not url:
            return None

        return {
            "author": author,
            "handle": handle,
            "text": text,
            "timestamp": timestamp,
            "url": url,
            "likes": likes,
            "retweets": retweets,
            "source_query": source_query,
        }
    except Exception:
        return None


# ── Tier 2: Google Cache ──────────────────────────────────────────────────────

def scrape_google(queries, max_results=50, delay=2.0):
    """
    Search Google for Twitter/X posts matching the queries and parse the
    result snippets.  No Playwright needed — plain HTTP with lxml/BeautifulSoup.

    Uses a minimum 5-second delay between Google requests to stay under the
    radar.
    """
    effective_delay = max(delay, 5.0)
    session = _make_session()
    results = []

    print(f"\n  {BOLD}[Google]{RESET} Searching for Twitter results via Google...\n")

    for query in queries:
        if len(results) >= max_results:
            break

        search_query = f"site:x.com OR site:twitter.com {query}"
        google_url = (
            f"https://www.google.com/search?q={quote_plus(search_query)}"
            f"&num=20&hl=en"
        )

        print(f"  Searching: {query}")
        resp = _retry_request(session, "GET", google_url, delay=effective_delay)
        if resp is None or not resp.ok:
            status = resp.status_code if resp else "no response"
            print(f"  {GRAY}Google returned {status} — skipping this query.{RESET}")
            time.sleep(_jitter(effective_delay))
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        entries = soup.select("div.g")

        for entry in entries:
            if len(results) >= max_results:
                break

            link_el = entry.select_one("a[href]")
            snippet_el = entry.select_one("div.VwiC3b, span.aCOpRe, div.s")
            title_el = entry.select_one("h3")

            if not link_el:
                continue

            href = link_el.get("href", "")
            if "twitter.com" not in href and "x.com" not in href:
                continue
            if "/status/" not in href:
                continue

            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            title_text = title_el.get_text(strip=True) if title_el else ""

            author, handle = _parse_google_author(title_text, href)

            results.append({
                "author": author,
                "handle": handle,
                "text": snippet,
                "timestamp": "",
                "url": href,
                "likes": 0,
                "retweets": 0,
                "source_query": query,
            })

        time.sleep(_jitter(effective_delay))

    return _deduplicate(results) if results else None


def _parse_google_author(title_text, url):
    """
    Best-effort extraction of author display name and handle from a Google
    result title or URL.

    Google titles for tweets often look like:
      'Display Name (@handle) / X'
      'Display Name on X: "tweet text"'
    """
    author = ""
    handle = ""

    if "(@" in title_text:
        parts = title_text.split("(@", 1)
        author = parts[0].strip()
        handle = parts[1].split(")")[0].strip() if ")" in parts[1] else ""
    elif " on X:" in title_text:
        author = title_text.split(" on X:")[0].strip()
    elif " on Twitter:" in title_text:
        author = title_text.split(" on Twitter:")[0].strip()

    if not handle:
        for prefix in ("https://twitter.com/", "https://x.com/",
                        "http://twitter.com/", "http://x.com/"):
            if url.startswith(prefix):
                remainder = url[len(prefix):]
                handle = remainder.split("/")[0]
                break

    return author, handle


# ── Tier 3: Nitter ────────────────────────────────────────────────────────────

def _find_live_nitter(session, timeout=5):
    """Ping known Nitter mirrors and return the first one that responds."""
    for mirror in NITTER_MIRRORS:
        try:
            resp = session.get(mirror, timeout=timeout, allow_redirects=True)
            if resp.ok:
                print(f"  {GREEN}Found live Nitter mirror:{RESET} {mirror}")
                return mirror
        except requests.exceptions.RequestException:
            continue
    return None


def scrape_nitter(queries, max_results=50, delay=2.0):
    """
    Scrape tweet search results from a live Nitter mirror.  Nitter renders
    server-side HTML, so we just need requests + lxml — no JavaScript.
    """
    session = _make_session()
    print(f"\n  {BOLD}[Nitter]{RESET} Looking for a live Nitter mirror...\n")

    base_url = _find_live_nitter(session)
    if not base_url:
        print(f"  {GRAY}No live Nitter mirrors found.{RESET}")
        return None

    results = []

    for query in queries:
        if len(results) >= max_results:
            break

        search_url = f"{base_url}/search?f=tweets&q={quote_plus(query)}"
        print(f"  Searching: {query}")
        resp = _retry_request(session, "GET", search_url, delay=delay)
        if resp is None or not resp.ok:
            time.sleep(_jitter(delay))
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        tweet_items = soup.select(".timeline-item, .tweet-body")

        for item in tweet_items:
            if len(results) >= max_results:
                break

            tweet = _parse_nitter_item(item, base_url, query)
            if tweet:
                results.append(tweet)

        time.sleep(_jitter(delay))

    return _deduplicate(results) if results else None


def _parse_nitter_item(item, base_url, source_query):
    """Extract tweet data from a Nitter HTML element."""
    try:
        fullname_el = item.select_one(".fullname")
        username_el = item.select_one(".username")
        author = fullname_el.get_text(strip=True) if fullname_el else ""
        handle = username_el.get_text(strip=True).lstrip("@") if username_el else ""

        content_el = item.select_one(".tweet-content, .media-body")
        text = content_el.get_text(" ", strip=True) if content_el else ""

        time_el = item.select_one("span.tweet-date a, time")
        timestamp = ""
        if time_el:
            timestamp = time_el.get("title", "") or time_el.get_text(strip=True)

        url = ""
        link_el = item.select_one("a.tweet-link, span.tweet-date a")
        if link_el:
            href = link_el.get("href", "")
            if href.startswith("/"):
                # Convert Nitter relative path to a twitter.com URL
                # Nitter path: /{handle}/status/{id}#m
                clean_href = href.split("#")[0]
                url = f"https://twitter.com{clean_href}"
            elif href.startswith("http"):
                url = href

        likes = 0
        retweets = 0
        stat_els = item.select(".tweet-stat, .icon-container")
        for stat in stat_els:
            stat_text = stat.get_text(strip=True)
            icon = stat.select_one(".icon-heart, .icon-retweet")
            if icon:
                class_list = " ".join(icon.get("class", []))
                if "heart" in class_list:
                    likes = _safe_int(stat_text)
                elif "retweet" in class_list:
                    retweets = _safe_int(stat_text)

        if not text and not url:
            return None

        return {
            "author": author,
            "handle": handle,
            "text": text,
            "timestamp": timestamp,
            "url": url,
            "likes": likes,
            "retweets": retweets,
            "source_query": source_query,
        }
    except Exception:
        return None


# ── Orchestration ─────────────────────────────────────────────────────────────

def scrape(queries, method="auto", max_results=50, scroll_count=15, delay=2.0):
    """
    Run the scraper using the specified method (or auto-cascade).
    Returns (results_list, method_name_used).
    """
    if method == "auto":
        methods_to_try = ["playwright", "google", "nitter"]
    else:
        methods_to_try = [method]

    for m in methods_to_try:
        print(f"  {BOLD}Trying method: {m}{RESET}")
        results = None

        if m == "playwright":
            results = scrape_playwright(
                queries, max_results=max_results,
                scroll_count=scroll_count, delay=delay,
            )

        elif m == "google":
            results = scrape_google(
                queries, max_results=max_results, delay=delay,
            )

        elif m == "nitter":
            results = scrape_nitter(
                queries, max_results=max_results, delay=delay,
            )

        else:
            print(f"  {GRAY}Unknown method '{m}' — skipping.{RESET}")
            continue

        if results:
            print(f"\n  {GREEN}Success with method: {m}{RESET}")
            return results, m

        print(f"  {GRAY}No results from {m} — falling back.{RESET}\n")

    return [], method


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Scrape Twitter/X for hackathon announcements (no API key needed).",
    )
    parser.add_argument(
        "-q", "--queries",
        type=str,
        default=",".join(DEFAULT_QUERIES),
        help="Comma-separated search terms (default: hackathon queries for SF/NYC/AI).",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "playwright", "google", "nitter"],
        default="auto",
        help="Scraping method: auto (cascade), playwright, google, or nitter.",
    )
    parser.add_argument(
        "-n", "--max-results",
        type=int,
        default=50,
        help="Maximum number of results to collect (default: 50).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to write JSON output file.",
    )
    parser.add_argument(
        "--scroll-count",
        type=int,
        default=15,
        help="Number of scroll iterations for Playwright (default: 15).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Base delay in seconds between requests (default: 2.0).",
    )
    return parser.parse_args(argv)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    print()
    print(f"  {BOLD}Twitter Hackathon Scraper{RESET}")
    print(f"  {GRAY}Queries:     {', '.join(queries)}{RESET}")
    print(f"  {GRAY}Method:      {args.method}{RESET}")
    print(f"  {GRAY}Max results: {args.max_results}{RESET}")
    print()

    results, method_used = scrape(
        queries,
        method=args.method,
        max_results=args.max_results,
        scroll_count=args.scroll_count,
        delay=args.delay,
    )

    _print_results(results)

    if args.output:
        output_data = _build_output(results, queries, method_used)
        try:
            with open(args.output, "w", encoding="utf-8") as fh:
                json.dump(output_data, fh, indent=2, ensure_ascii=False)
            print(f"  {GREEN}Results written to {args.output}{RESET}\n")
        except OSError as exc:
            print(f"  {GRAY}Failed to write output file: {exc}{RESET}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
