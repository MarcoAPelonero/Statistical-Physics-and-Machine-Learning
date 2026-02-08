#!/usr/bin/env python3
"""
WORKING PubMed dataset fetcher (titles + publication data + MeSH) using:
- History server (usehistory=y) per time-slice
- Small per-slice retrieval from retstart=0 only (NO deep offsets, NO random retstart)
- Approximate sampling: take the first K records per month (or enough to hit global N)

This avoids:
- ESearch 10k ID window problems
- EFetch 400 errors from huge retstart jumps

Outputs JSONL by default (one JSON object per line). Optional JSON array.

Example:
  python datafetch.py -n 10000 --start-year 2000 --end-year 2019 \
    --email name@provider.com --out dataset.jsonl --english-only --chunk 200 --min-mesh 8

Notes:
- This is NOT “true randomness”. It is deterministic-ish within each month because we take from the head.
- You explicitly said you do not care about true randomness; you want a functioning dataset fetcher.
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"

# No-key guidance: <= 3 req/s
MIN_INTERVAL_S = (1.0 / 3.0) + 0.03  # ~0.363 s

MAX_IDS_PER_EFETCH_GET = 200
MAX_IDS_PER_ESEARCH_PAGE = 200


class RateLimiter:
    def __init__(self, min_interval_s: float):
        self.min_interval_s = float(min_interval_s)
        self._last_t: Optional[float] = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last_t is None:
            self._last_t = now
            return
        elapsed = now - self._last_t
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_t = time.monotonic()


def _http_get(
    session: requests.Session,
    limiter: RateLimiter,
    url: str,
    params: Dict[str, Any],
    timeout: float,
    max_retries: int,
    backoff: float,
) -> requests.Response:
    attempt = 0
    while True:
        attempt += 1
        limiter.wait()
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504) and attempt <= max_retries:
                mult = 2.0 if resp.status_code == 429 else 1.0
                time.sleep(mult * backoff * (2 ** (attempt - 1)))
                continue
            body = (resp.text or "")[:500].replace("\n", " ")
            raise requests.HTTPError(f"{resp.status_code} for {resp.url} | body: {body}", response=resp)
        except requests.RequestException:
            if attempt > max_retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))


def _ymd(d: date) -> str:
    return d.strftime("%Y/%m/%d")


def build_base_term(english_only: bool) -> str:
    term = 'medline[sb] AND "journal article"[pt]'
    if english_only:
        term += " AND english[la]"
    return term


def term_with_pdat_range(base_term: str, mindate: date, maxdate: date) -> str:
    return f'({base_term}) AND ("{_ymd(mindate)}"[pdat] : "{_ymd(maxdate)}"[pdat])'


def month_slices(start_year: int, end_year: int) -> List[Tuple[date, date]]:
    out: List[Tuple[date, date]] = []
    cur = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    while cur <= end:
        last_day = calendar.monthrange(cur.year, cur.month)[1]
        a = date(cur.year, cur.month, 1)
        b = date(cur.year, cur.month, last_day)
        out.append((a, b))
        # next month
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return out


def esearch_history(
    session: requests.Session,
    limiter: RateLimiter,
    term: str,
    tool: str,
    email: str,
    timeout: float,
    max_retries: int,
    backoff: float,
) -> Tuple[int, str, str]:
    """
    ESearch with usehistory=y returns Count, WebEnv, QueryKey.
    """
    params: Dict[str, Any] = {
        "db": "pubmed",
        "term": term,
        "retmode": "xml",
        "retmax": 0,
        "usehistory": "y",
        "tool": tool,
        "email": email,
    }
    resp = _http_get(session, limiter, ESEARCH_URL, params, timeout, max_retries, backoff)
    root = ET.fromstring(resp.text)

    count = int(root.findtext("./Count", default="0"))
    webenv = root.findtext("./WebEnv", default="").strip()
    query_key = root.findtext("./QueryKey", default="").strip()
    if count > 0 and (not webenv or not query_key):
        raise RuntimeError("ESearch returned results but missing WebEnv/QueryKey.")
    return count, webenv, query_key


def efetch_history_batch(
    session: requests.Session,
    limiter: RateLimiter,
    webenv: str,
    query_key: str,
    retstart: int,
    retmax: int,
    tool: str,
    email: str,
    timeout: float,
    max_retries: int,
    backoff: float,
) -> str:
    """
    Fetch batch from a history set. IMPORTANT: we only ever use small retstart (0, 200, 400, ...)
    within each monthly slice, never millions.
    """
    retmax = min(int(retmax), MAX_IDS_PER_EFETCH_GET, MAX_IDS_PER_ESEARCH_PAGE)
    params: Dict[str, Any] = {
        "db": "pubmed",
        "WebEnv": webenv,
        "query_key": query_key,
        "retstart": int(retstart),
        "retmax": int(retmax),
        "retmode": "xml",
        "tool": tool,
        "email": email,
    }
    resp = _http_get(session, limiter, EFETCH_URL, params, timeout, max_retries, backoff)
    return resp.text


def _safe_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    x = x.strip()
    if not x:
        return None
    try:
        return int(x)
    except ValueError:
        return None


def _extract_year_from_pubdate(pubdate_elem: ET.Element) -> Optional[int]:
    y = pubdate_elem.findtext("Year")
    if y:
        yi = _safe_int(y)
        if yi is not None:
            return yi

    md = pubdate_elem.findtext("MedlineDate")
    if md:
        m = re.search(r"\b(19|20)\d{2}\b", md)
        if m:
            return _safe_int(m.group(0))
    return None


def _parse_pubmed_batch(
    xml_text: str,
    major_only: bool,
) -> List[Dict[str, Any]]:
    """
    Parse PubMedArticleSet XML into dict records with:
    pmid, title, pub_year, pub_month, pub_day, journal, volume, issue, mesh[]
    """
    root = ET.fromstring(xml_text)
    out: List[Dict[str, Any]] = []

    for pubmed_article in root.findall("./PubmedArticle"):
        medline = pubmed_article.find("./MedlineCitation")
        if medline is None:
            continue

        pmid = medline.findtext("./PMID")
        if not pmid:
            continue
        pmid = pmid.strip()

        article = medline.find("./Article")
        title: Optional[str] = None
        journal_title: Optional[str] = None
        volume: Optional[str] = None
        issue: Optional[str] = None
        pub_year: Optional[int] = None
        pub_month: Optional[str] = None
        pub_day: Optional[str] = None

        if article is not None:
            t = article.findtext("./ArticleTitle")
            if t:
                title = t.strip()

            j = article.find("./Journal")
            if j is not None:
                journal_title = j.findtext("./Title")
                if journal_title:
                    journal_title = journal_title.strip()

                ji = j.find("./JournalIssue")
                if ji is not None:
                    volume = ji.findtext("./Volume")
                    if volume:
                        volume = volume.strip()
                    issue = ji.findtext("./Issue")
                    if issue:
                        issue = issue.strip()

                    pd = ji.find("./PubDate")
                    if pd is not None:
                        pub_year = _extract_year_from_pubdate(pd)
                        m = pd.findtext("./Month")
                        d = pd.findtext("./Day")
                        if m:
                            pub_month = m.strip()
                        if d:
                            pub_day = d.strip()

        # MeSH descriptors
        mesh_terms: List[Dict[str, Any]] = []
        mesh_list = medline.find("./MeshHeadingList")
        if mesh_list is not None:
            for mh in mesh_list.findall("./MeshHeading"):
                d = mh.find("./DescriptorName")
                if d is None or d.text is None:
                    continue
                ui = (d.attrib.get("UI", "") or "").strip()
                name = d.text.strip()
                major = (d.attrib.get("MajorTopicYN", "N").upper() == "Y")
                if major_only and not major:
                    continue
                mesh_terms.append({"ui": ui, "name": name, "major": major})

        out.append(
            {
                "pmid": pmid,
                "title": title,
                "pub_year": pub_year,
                "pub_month": pub_month,
                "pub_day": pub_day,
                "journal": journal_title,
                "volume": volume,
                "issue": issue,
                "mesh": mesh_terms,
            }
        )

    return out


def write_record(out_fp, fmt: str, rec: Dict[str, Any], first_json: bool) -> bool:
    if fmt == "jsonl":
        out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return first_json
    # json array
    if not first_json:
        out_fp.write(",\n")
    out_fp.write(json.dumps(rec, ensure_ascii=False))
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", "-n", type=int, required=True)
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--end-year", type=int, required=True)
    ap.add_argument("--email", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tool", default="pubmed-month-sampler-history")

    ap.add_argument("--english-only", action="store_true")
    ap.add_argument("--major-only", action="store_true")
    ap.add_argument("--min-mesh", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=200, help="EFetch batch size (<=200).")
    ap.add_argument("--format", choices=("jsonl", "json"), default="jsonl")

    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--backoff", type=float, default=0.5)

    # If you want "first K/month" behavior explicitly, set this; otherwise it auto-allocates to hit N.
    ap.add_argument("--per-month", type=int, default=0, help="If >0, try to take this many from each month.")

    args = ap.parse_args()

    if args.start_year > args.end_year:
        raise SystemExit("start-year must be <= end-year")
    if args.n <= 0:
        raise SystemExit("--n must be > 0")
    if args.min_mesh < 0:
        raise SystemExit("--min-mesh must be >= 0")

    chunk = min(int(args.chunk), MAX_IDS_PER_EFETCH_GET, MAX_IDS_PER_ESEARCH_PAGE)
    if chunk <= 0:
        raise SystemExit("--chunk must be > 0")

    base_term = build_base_term(english_only=args.english_only)

    limiter = RateLimiter(MIN_INTERVAL_S)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": f"{args.tool} ({args.email})",
            "Accept-Encoding": "gzip, deflate",
        }
    )

    slices = month_slices(args.start_year, args.end_year)
    total_months = len(slices)

    out_fp = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    try:
        if args.format == "json":
            out_fp.write("[\n")

        seen_pmids = set()
        written = 0
        first_json = True

        # Month-level progress bar (each month has an ESearch + some EFetch calls)
        p_months = tqdm(total=total_months, desc="Months", unit="month", file=sys.stderr, position=0, leave=True)
        # Record-level progress bar
        p_write = tqdm(total=args.n, desc="Writing", unit="article", file=sys.stderr, position=1, leave=True)

        try:
            for idx, (a, b) in enumerate(slices):
                if written >= args.n:
                    break

                # Target for this month:
                if args.per_month > 0:
                    month_target = args.per_month
                else:
                    # dynamic: spread remaining across remaining months
                    remaining = args.n - written
                    months_left = total_months - idx
                    month_target = max(1, remaining // months_left)
                    # give some slack so we don't starve early months
                    # (optional heuristic; keeps flow stable)
                    month_target = int(min(month_target * 1.2, remaining))

                q = term_with_pdat_range(base_term, a, b)

                # ESearch history for this month
                count, webenv, query_key = esearch_history(
                    session=session,
                    limiter=limiter,
                    term=q,
                    tool=args.tool,
                    email=args.email,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    backoff=args.backoff,
                )

                # nothing in this month
                if count <= 0:
                    p_months.update(1)
                    continue

                # Fetch pages from retstart=0 upward until we fill month_target or run out
                collected_this_month = 0
                retstart = 0

                # inner progress for this month (optional)
                # We cap to month_target, but show some feedback.
                p_inner = tqdm(
                    total=min(month_target, count),
                    desc=f"Fetch {a.strftime('%Y-%m')}",
                    unit="article",
                    file=sys.stderr,
                    position=2,
                    leave=False,
                )

                try:
                    while retstart < count and written < args.n and collected_this_month < month_target:
                        xml_text = efetch_history_batch(
                            session=session,
                            limiter=limiter,
                            webenv=webenv,
                            query_key=query_key,
                            retstart=retstart,
                            retmax=chunk,
                            tool=args.tool,
                            email=args.email,
                            timeout=args.timeout,
                            max_retries=args.max_retries,
                            backoff=args.backoff,
                        )
                        recs = _parse_pubmed_batch(xml_text, major_only=args.major_only)
                        if not recs:
                            # advance to avoid infinite loops
                            retstart += chunk
                            continue

                        for rec in recs:
                            if written >= args.n or collected_this_month >= month_target:
                                break

                            pmid = rec.get("pmid")
                            if not pmid or pmid in seen_pmids:
                                continue
                            if args.min_mesh and len(rec.get("mesh", [])) < args.min_mesh:
                                continue

                            seen_pmids.add(pmid)

                            first_json = write_record(out_fp, args.format, rec, first_json)

                            written += 1
                            collected_this_month += 1
                            p_write.update(1)
                            p_inner.update(1)

                        retstart += chunk

                    p_write.set_postfix(
                        {
                            "written": written,
                            "month": a.strftime("%Y-%m"),
                            "month_ok": collected_this_month,
                            "seen": len(seen_pmids),
                        }
                    )

                finally:
                    p_inner.close()

                p_months.update(1)

            if args.format == "json":
                out_fp.write("\n]\n")

            if written < args.n:
                print(
                    f"WARNING: only produced {written}/{args.n}. "
                    f"Try lowering --min-mesh, dropping --major-only, widening range, or set --per-month to 0.",
                    file=sys.stderr,
                )

        finally:
            p_write.close()
            p_months.close()

    finally:
        if out_fp is not sys.stdout:
            out_fp.close()


if __name__ == "__main__":
    main()
