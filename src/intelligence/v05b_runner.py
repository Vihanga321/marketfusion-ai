"""Run MarketFusion V0.5B continuous free news + macro intelligence."""
from __future__ import annotations

import argparse
import time

import pandas as pd

from .features import build_context_snapshot
from .macro_collectors import collect_macro
from .news_collectors import collect_news
from .quality import build_quality_report, write_quality_report
from .store import (
    append_context, atomic_write_json, load_stores, merge_macro, merge_news, save_stores,
)
from .v05b_contract import CONTRACT_VERSION, DEFAULT_INTERVAL_SECONDS


def run_cycle() -> dict:
    captured_at = pd.Timestamp.now(tz="UTC")
    existing_news, existing_macro, existing_context = load_stores()

    incoming_news, news_errors, news_counts = collect_news(captured_at)
    incoming_macro, macro_errors, macro_counts = collect_macro(captured_at)
    news, new_news = merge_news(existing_news, incoming_news)
    macro, new_macro = merge_macro(existing_macro, incoming_macro)
    current_context = build_context_snapshot(news, macro, captured_at)
    quality = build_quality_report(news, macro, current_context, captured_at, news_errors, macro_errors)
    write_quality_report(quality)

    context = existing_context
    if quality.passed:
        context = append_context(existing_context, current_context)
        save_stores(news, macro, context)
    else:
        # Preserve successfully captured canonical source data even when the
        # feature gate fails, but do not append a model-visible context row.
        from .store import atomic_write_parquet
        from .v05b_contract import MACRO_FILE, NEWS_FILE
        atomic_write_parquet(news, NEWS_FILE)
        atomic_write_parquet(macro, MACRO_FILE)

    degraded = bool(news_errors or macro_errors)
    status = "PASS_RUNNING_DEGRADED" if quality.passed and degraded else ("PASS_RUNNING" if quality.passed else "FAIL_CLOSED")
    latest_context = current_context.iloc[0].to_dict() if not current_context.empty else {}
    payload = {
        "contract_version": CONTRACT_VERSION,
        "status": status,
        "captured_at_utc": captured_at.isoformat(),
        "new_news_rows": int(new_news),
        "news_rows_total": int(len(news)),
        "new_macro_rows": int(new_macro),
        "macro_rows_total": int(len(macro)),
        "context_rows_total": int(len(context)),
        "news_source_counts": news_counts,
        "macro_source_counts": macro_counts,
        "news_errors": news_errors,
        "macro_errors": macro_errors,
        "quality_failures": quality.failures,
        "current_news_15m": int(latest_context.get("news_15m_count", 0) or 0),
        "current_news_60m": int(latest_context.get("news_60m_count", 0) or 0),
        "current_news_240m": int(latest_context.get("news_240m_count", 0) or 0),
        "current_news_1440m": int(latest_context.get("news_1440m_count", 0) or 0),
        "trading": "DISABLED / NOT IMPLEMENTED",
    }
    for key in (
        "macro_us_effective_fed_funds_value", "macro_us_2y_yield_value",
        "macro_us_10y_yield_value", "macro_us_10y_minus_2y_value",
        "macro_ecb_main_refinancing_rate_value",
        "macro_policy_rate_spread_us_minus_ecb_pctpt",
    ):
        payload[key] = latest_context.get(key)
    atomic_write_json(payload)

    print("MARKETFUSION V0.5B CONTINUOUS NEWS + MACRO")
    print(f"status: {status}")
    print(f"captured_at_utc: {captured_at.isoformat()}")
    print(f"news: total={len(news)} new={new_news} source_errors={len(news_errors)}")
    print(f"macro: total={len(macro)} new={new_macro} source_errors={len(macro_errors)}")
    print(f"context_history_rows: {len(context)}")
    print(f"news_windows: 15m={payload['current_news_15m']} 60m={payload['current_news_60m']} 240m={payload['current_news_240m']} 24h={payload['current_news_1440m']}")
    print(f"fed_funds: {payload.get('macro_us_effective_fed_funds_value')}")
    print(f"us_2y: {payload.get('macro_us_2y_yield_value')}")
    print(f"us_10y: {payload.get('macro_us_10y_yield_value')}")
    print(f"ecb_mrr: {payload.get('macro_ecb_main_refinancing_rate_value')}")
    print("availability: first MarketFusion capture; no retrospective backdating")
    print("trading: DISABLED / NOT IMPLEMENTED")
    print(f"V05B_STATUS: {'PASS' if quality.passed else 'FAIL'}")
    if news_errors:
        print("NEWS_WARNINGS:")
        for error in news_errors:
            print(f"- {error}")
    if macro_errors:
        print("MACRO_WARNINGS:")
        for error in macro_errors:
            print(f"- {error}")
    return payload


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.once:
        result = run_cycle()
        if str(result["status"]).startswith("FAIL"):
            raise SystemExit(1)
        return
    interval = max(60, int(args.interval))
    print(f"Starting V0.5B continuous intelligence every {interval}s. Ctrl+C to stop.")
    while True:
        try:
            run_cycle()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"V0.5B cycle error: {type(exc).__name__}: {exc}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
