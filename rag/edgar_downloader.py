"""SEC EDGAR REST API client — CIK lookup + filing download."""

import time
import logging
from pathlib import Path

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": settings.edgar_user_agent}
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{filename}"
# Fallback: directory listing (returns HTML, not JSON — parsed separately)
_DIRECTORY_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={filing_type}&dateb=&owner=include&count=1&search_text=&output=atom"
_RATE_SLEEP = 0.12  # stay under 10 req/sec


def _get(url: str) -> requests.Response:
    time.sleep(_RATE_SLEEP)
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


class EDGARDownloader:
    def __init__(self):
        self._cik_cache: dict[str, str] = {}
        self._submissions_cache: dict[str, dict] = {}

    def get_cik_for_ticker(self, ticker: str) -> str:
        """Return zero-padded 10-digit CIK string for ticker."""
        ticker_upper = ticker.upper()
        if ticker_upper in self._cik_cache:
            return self._cik_cache[ticker_upper]

        data = _get(_TICKERS_URL).json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik_str = str(entry["cik_str"]).zfill(10)
                self._cik_cache[ticker_upper] = cik_str
                return cik_str

        raise ValueError(f"Ticker {ticker} not found in EDGAR company_tickers.json")

    def _get_submissions(self, cik: str) -> dict:
        if cik not in self._submissions_cache:
            url = _SUBMISSIONS_URL.format(cik=int(cik))
            self._submissions_cache[cik] = _get(url).json()
        return self._submissions_cache[cik]

    def get_latest_filing(self, cik: str, filing_type: str) -> tuple[str, str]:
        """
        Return (accession_number, primary_document_filename) for the most recent filing.

        The submissions JSON includes primaryDocument directly — no index fetch needed.
        For large filers with many filings, also checks older filings pages.
        """
        submissions = self._get_submissions(cik)
        company_name = submissions.get("name", "")

        pages_to_check = [submissions.get("filings", {}).get("recent", {})]

        # EDGAR paginates older filings into separate JSON files
        for extra in submissions.get("filings", {}).get("files", []):
            extra_url = f"https://data.sec.gov/submissions/{extra['name']}"
            try:
                pages_to_check.append(_get(extra_url).json())
            except Exception:
                pass

        for page in pages_to_check:
            forms = page.get("form", [])
            accessions = page.get("accessionNumber", [])
            primary_docs = page.get("primaryDocument", [])

            for form, accession, primary_doc in zip(forms, accessions, primary_docs):
                if form == filing_type and primary_doc:
                    logger.info("Found %s filing: accession=%s doc=%s", filing_type, accession, primary_doc)
                    return accession, primary_doc

        raise ValueError(f"No {filing_type} filing found for CIK {cik}")

    def _download_document(self, cik: str, accession: str, filename: str) -> str:
        accession_nodash = accession.replace("-", "")
        url = _DOCUMENT_URL.format(
            cik=int(cik), accession_nodash=accession_nodash, filename=filename
        )
        return _get(url).text

    def download_filing(
        self,
        ticker: str,
        filing_type: str = "10-K",
    ) -> tuple[str, str, str, str]:
        """
        Full pipeline: ticker → download filing HTML.

        Returns (cik, company_name, accession, local_path).
        """
        cik = self.get_cik_for_ticker(ticker)
        submissions = self._get_submissions(cik)
        company_name = submissions.get("name", ticker)

        accession, primary_doc = self.get_latest_filing(cik, filing_type)
        out_dir = Path(settings.sec_filings_dir) / ticker.upper() / filing_type
        out_dir.mkdir(parents=True, exist_ok=True)
        local_path = out_dir / primary_doc

        if local_path.exists():
            logger.info("Using cached filing: %s", local_path)
            return cik, company_name, accession, str(local_path)

        content = self._download_document(cik, accession, primary_doc)
        local_path.write_text(content, encoding="utf-8")

        logger.info(
            "Downloaded %s %s → %s (%d chars)", ticker, filing_type, local_path, len(content)
        )
        return cik, company_name, accession, str(local_path)
