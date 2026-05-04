"""CLI fallback for the Financial Research Agent.

Usage:
    python cli.py AAPL
    python cli.py AAPL --mode stock
    python cli.py AAPL --mode sec --filing-type 10-Q
    python cli.py AAPL --mode combined
"""

import sys
import json
import logging
import argparse
from typing import Literal

# Windows terminals default to cp1252 which can't print ⚠️ / ✓ etc.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Financial Research Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL, TSLA, MSFT)")
    parser.add_argument(
        "--mode",
        choices=["stock", "sec", "combined"],
        default="stock",
        help="Analysis mode: stock, sec, or combined (default: stock)",
    )
    parser.add_argument(
        "--filing-type",
        dest="filing_type",
        choices=["10-K", "10-Q"],
        default="10-K",
        help="SEC filing type for sec/combined modes (default: 10-K)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output structured JSON instead of markdown",
    )
    parser.add_argument(
        "--no-guardrails",
        action="store_true",
        dest="no_guardrails",
        help="Skip guardrails (for testing)",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper().strip()
    mode: Literal["stock", "sec", "combined"] = args.mode

    print(f"\nResearching {ticker} ({mode} mode)...\n")

    if args.no_guardrails:
        _run_direct(ticker, mode, args.filing_type, args.json_output)
    else:
        _run_with_guardrails(ticker, mode, args.filing_type, args.json_output)


def _run_direct(
    ticker: str,
    mode: str,
    filing_type: str,
    json_output: bool,
) -> None:
    from graph.supervisor import build_supervisor_graph, build_initial_supervisor_state

    graph = build_supervisor_graph()
    state = build_initial_supervisor_state(ticker, mode, filing_type)

    result = graph.invoke(state)

    if json_output:
        output = result.get("structured_output") or {
            "ticker": ticker,
            "mode": mode,
            "report": result.get("comprehensive_report", "")[:500],
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        report = result.get("comprehensive_report") or result.get("investment_brief", "No report generated.")
        print(report)


def _run_with_guardrails(
    ticker: str,
    mode: str,
    filing_type: str,
    json_output: bool,
) -> None:
    from guardrails.middleware import run_with_guardrails

    query = f"Research {ticker}"
    final_response, metadata = run_with_guardrails(
        user_input=query,
        ticker=ticker,
        mode=mode,
        filing_type=filing_type,
    )

    if metadata.get("blocked"):
        print(f"[BLOCKED] {final_response}")
        sys.exit(1)

    if json_output:
        print(json.dumps({
            "ticker": ticker,
            "mode": mode,
            "session_id": metadata.get("session_id"),
            "report_preview": final_response[:500],
        }, indent=2))
    else:
        print(final_response)


if __name__ == "__main__":
    main()
