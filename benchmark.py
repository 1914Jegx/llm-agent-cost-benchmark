#!/usr/bin/env python3
"""Cost, latency and rule-check benchmark for a customer-service LLM agent.

Sends a fixed set of synthetic WhatsApp conversations (a fictional camera and
alarm store, see data/) to several models through OpenRouter, measures latency,
tokens and cost of every reply, runs cheap automatic checks on each reply, and
writes one CSV row per reply plus a per-model summary CSV.

This is the public, simplified version of the method described in README.md:
one model call per case, no tools, no LLM judge. Read the replies before you
trust a ranking: the checks are heuristics, not a grade.

Requires OPENROUTER_API_KEY for real runs. --dry-run needs no key and no network.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:  # --help and --dry-run still work without it
    requests = None

ROOT = Path(__file__).resolve().parent
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# OpenRouter model IDs as of October 2026. Model IDs change: check
# https://openrouter.ai/models and pass --models to override.
DEFAULT_MODELS = [
    "anthropic/claude-haiku-5.5",
    "deepseek/deepseek-v4.1-flash",
    "google/gemini-3.5-flash-lite",
    "z-ai/glm-5.3-flash",
    "openai/gpt-6-luna",
    "qwen/qwen3.7-flash",
]

# A money figure: "$129.900", "$ 1.149.000", "129.900", "$129900", "$1,149,000".
# Bare numbers only count when they carry thousand separators, so years, hours,
# "4 MP" or "128 GB" are not mistaken for prices.
FIGURE_RE = re.compile(r"(?<![\d.,])(\$\s?)?(\d{1,3}(?:[.,]\d{3})+|\d{4,})(?![\d])")

CSV_FIELDS = [
    "run_at",
    "model",
    "provider",
    "case_id",
    "category",
    "repeat",
    "latency_s",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cost_usd",
    "finish_reason",
    "reply_chars",
    "figures",
    "unlisted_figures",
    "failed_checks",
    "checks_passed",
    "error",
    "reply",
]

SUMMARY_FIELDS = [
    "model",
    "replies",
    "errors",
    "checks_passed",
    "replies_with_unlisted_figures",
    "usd_per_reply",
    "usd_total",
    "latency_median_s",
    "latency_p90_s",
    "prompt_cache_hit_pct",
    "avg_completion_tokens",
    "avg_reply_chars",
]


# ---------------------------------------------------------------- data ----


def cop(amount: int) -> str:
    """Format pesos the Colombian way: 129900 -> $129.900."""
    return "$" + f"{amount:,}".replace(",", ".")


def load_store(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_system_prompt(template: str, store: dict) -> str:
    catalog = "\n".join(
        f"- {p['sku']} | {p['name']} | {cop(p['price'])} | existencias: {p['stock']}"
        for p in store["catalog"]
    )
    city, *rest = store["metro_area"]
    metro = f"el área metropolitana de {city} ({', '.join(rest[:-1])} y {rest[-1]})"
    fees = store["fees"]
    return template.format(
        store_name=store["name"],
        city=store["city"],
        catalog=catalog,
        metro_area=metro,
        shipping=cop(fees["shipping"]),
        free_shipping_from=cop(fees["free_shipping_from"]),
        install_per_camera=cop(fees["install_per_camera"]),
        install_alarm_kit=cop(fees["install_alarm_kit"]),
        canary=store["canary"],
    )


def allowed_figures(store: dict) -> set[int]:
    """Every figure a correct reply could contain: catalog prices and fees times 1..N."""
    fees = store["fees"]
    unit_amounts = [p["price"] for p in store["catalog"]] + [
        fees["shipping"],
        fees["install_per_camera"],
        fees["install_alarm_kit"],
    ]
    max_q = store.get("max_quantity_for_totals", 10)
    allowed = {amount * q for amount in unit_amounts for q in range(1, max_q + 1)}
    allowed.add(fees["free_shipping_from"])
    return allowed


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    for case in cases:
        cid = case.get("id")
        if not cid or cid in seen:
            raise ValueError(f"case id missing or duplicated: {cid!r}")
        seen.add(cid)
        messages = case.get("messages") or []
        if not messages or messages[-1].get("role") != "user":
            raise ValueError(f"{cid}: the last message must come from the user")
        checks = case.get("checks") or {}
        for pattern in checks.get("must_match", []) + checks.get("must_not_match", []):
            re.compile(pattern)  # fail early on a broken pattern
    return cases


# -------------------------------------------------------------- checks ----


def extract_figures(text: str) -> list[int]:
    figures = []
    for match in FIGURE_RE.finditer(text):
        has_currency, digits = match.group(1), match.group(2)
        if not has_currency and not re.search(r"[.,]", digits):
            continue
        value = int(re.sub(r"[.,]", "", digits))
        if value >= 1000:
            figures.append(value)
    return figures


def run_checks(reply: str, case: dict, base_allowed: set[int]) -> dict:
    checks = case.get("checks") or {}
    failed = []
    for pattern in checks.get("must_match", []):
        if not re.search(pattern, reply, re.IGNORECASE):
            failed.append(f"must_match /{pattern}/")
    for pattern in checks.get("must_not_match", []):
        if re.search(pattern, reply, re.IGNORECASE):
            failed.append(f"must_not_match /{pattern}/")
    max_chars = checks.get("max_chars")
    if max_chars and len(reply) > max_chars:
        failed.append(f"max_chars {len(reply)}>{max_chars}")
    allowed = base_allowed | set(checks.get("allowed_figures", []))
    figures = extract_figures(reply)
    unlisted = sorted({f for f in figures if f not in allowed})
    return {"figures": figures, "unlisted": unlisted, "failed": failed}


# ----------------------------------------------------------------- API ----


def call_model(session, api_key: str, model: str, messages: list[dict], args) -> dict:
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if not args.allow_data_collection:
        # Route only to providers that do not store or train on prompts.
        body["provider"] = {"data_collection": "deny"}
    if args.reasoning == "off":
        body["reasoning"] = {"enabled": False}
    elif args.reasoning in ("low", "medium", "high"):
        body["reasoning"] = {"effort": args.reasoning}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    result = {"reply": "", "error": ""}
    for attempt in range(args.retries + 1):
        started = time.perf_counter()
        try:
            response = session.post(API_URL, json=body, headers=headers, timeout=args.timeout)
            latency = time.perf_counter() - started
        except requests.RequestException as exc:
            result["error"] = f"network: {exc}"
            if attempt < args.retries:
                time.sleep(2**attempt * 2)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
            if attempt < args.retries:
                time.sleep(2**attempt * 2)
            continue
        try:
            data = response.json()
        except ValueError:
            return {**result, "error": f"HTTP {response.status_code}: non-JSON body"}
        if response.status_code != 200 or "error" in data:
            message = (data.get("error") or {}).get("message", response.text[:200])
            return {**result, "error": f"HTTP {response.status_code}: {message}"}

        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return {
            "reply": ((choice.get("message") or {}).get("content") or "").strip(),
            "finish_reason": choice.get("finish_reason", ""),
            "provider": data.get("provider", ""),
            "latency_s": round(latency, 3),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens", 0
            ),
            # OpenRouter returns the billed cost of every response in usage.cost.
            "cost_usd": usage.get("cost"),
            "error": "",
        }
    return result


# ------------------------------------------------------------- summary ----


def p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for model in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == model]
        ok = [r for r in mine if not r["error"]]
        costs = [float(r["cost_usd"]) for r in ok if r["cost_usd"] not in ("", None)]
        latencies = [float(r["latency_s"]) for r in ok]
        prompt = sum(int(r["prompt_tokens"] or 0) for r in ok)
        cached = sum(int(r["cached_tokens"] or 0) for r in ok)
        completion = [int(r["completion_tokens"] or 0) for r in ok]
        chars = [int(r["reply_chars"]) for r in ok]
        summary.append(
            {
                "model": model,
                "replies": len(ok),
                "errors": len(mine) - len(ok),
                "checks_passed": f"{sum(1 for r in ok if r['checks_passed'])}/{len(ok)}",
                "replies_with_unlisted_figures": sum(1 for r in ok if r["unlisted_figures"]),
                "usd_per_reply": f"{statistics.mean(costs):.6f}" if costs else "",
                "usd_total": f"{sum(costs):.6f}" if costs else "",
                "latency_median_s": f"{statistics.median(latencies):.2f}" if latencies else "",
                "latency_p90_s": f"{p90(latencies):.2f}" if latencies else "",
                "prompt_cache_hit_pct": f"{100 * cached / prompt:.0f}" if prompt else "",
                "avg_completion_tokens": f"{statistics.mean(completion):.0f}" if ok else "",
                "avg_reply_chars": f"{statistics.mean(chars):.0f}" if ok else "",
            }
        )
    return summary


def print_table(rows: list[dict], fields: list[str]) -> None:
    widths = {f: max(len(f), *(len(str(r[f])) for r in rows)) for f in fields}
    print("  ".join(f.ljust(widths[f]) for f in fields))
    for row in rows:
        print("  ".join(str(row[f]).ljust(widths[f]) for f in fields))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- main ----


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LLMs as a customer-service agent: cost, latency, tokens "
        "and automatic rule checks over synthetic conversations, via OpenRouter.",
        epilog="Example: python benchmark.py --models anthropic/claude-haiku-5.5,"
        "openai/gpt-6-luna --max-cost 0.25",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="comma-separated OpenRouter model IDs (default: %(default)s)",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data" / "cases.json",
        help="conversation cases (default: data/cases.json)",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=ROOT / "data" / "store.json",
        help="fictional catalog and fees (default: data/store.json)",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=ROOT / "data" / "system_prompt.txt",
        help="system prompt template (default: data/system_prompt.txt)",
    )
    parser.add_argument("--only", default="", help="comma-separated case IDs to run")
    parser.add_argument(
        "--repeat", type=int, default=1, help="times each case is sent to each model (default: 1)"
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=1.0,
        help="stop when the run has spent this many US$ (default: 1.00)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=600, help="max completion tokens per reply (default: 600)"
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="(default: 0.3)")
    parser.add_argument(
        "--reasoning",
        choices=["default", "off", "low", "medium", "high"],
        default="default",
        help="'default' sends nothing; some models reject 'off' because "
        "reasoning is mandatory for them (default: default)",
    )
    parser.add_argument(
        "--allow-data-collection",
        action="store_true",
        help="also route to providers that may store or train on prompts",
    )
    parser.add_argument("--timeout", type=float, default=120, help="seconds per request")
    parser.add_argument("--retries", type=int, default=2, help="retries on HTTP 429/5xx")
    parser.add_argument(
        "--out", type=Path, default=None, help="output CSV (default: runs/<UTC timestamp>.csv)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the data and print the plan; no network, no key",
    )
    parser.add_argument(
        "--show-prompt", action="store_true", help="print the rendered system prompt and exit"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = load_store(args.store)
    system_prompt = render_system_prompt(args.prompt.read_text(encoding="utf-8"), store)
    if args.show_prompt:
        print(system_prompt)
        return 0

    cases = load_cases(args.cases)
    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        unknown = wanted - {c["id"] for c in cases}
        if unknown:
            print(f"unknown case IDs: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c["id"] in wanted]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    base_allowed = allowed_figures(store)
    planned = len(models) * len(cases) * args.repeat

    print(
        f"{len(cases)} cases x {len(models)} models x {args.repeat} = {planned} calls; "
        f"system prompt {len(system_prompt)} chars; spend cap US${args.max_cost:g}"
    )
    if args.dry_run:
        for case in cases:
            print(f"  {case['id']:<28} {case['category']:<14} {case['note']}")
        print("models: " + ", ".join(models))
        print("dry run: data is valid, nothing was sent")
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print(
            "OPENROUTER_API_KEY is not set (use --dry-run to validate without it)", file=sys.stderr
        )
        return 2
    if requests is None:
        print("the 'requests' package is missing: pip install -r requirements.txt", file=sys.stderr)
        return 2

    run_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or ROOT / "runs" / f"{run_at}.csv"
    rows: list[dict] = []
    spent = 0.0
    session = requests.Session()
    stopped = False
    # Model-major order keeps each model's calls back to back, so its prompt
    # cache can warm up the way it would in production.
    for model in models:
        for rep in range(1, args.repeat + 1):
            for case in cases:
                if spent >= args.max_cost:
                    stopped = True
                    break
                messages = [{"role": "system", "content": system_prompt}, *case["messages"]]
                res = call_model(session, api_key, model, messages, args)
                spent += float(res.get("cost_usd") or 0)
                reply = res.get("reply", "")
                check = (
                    run_checks(reply, case, base_allowed)
                    if reply
                    else {"figures": [], "unlisted": [], "failed": ["empty reply"]}
                )
                error = res.get("error", "")
                rows.append(
                    {
                        "run_at": run_at,
                        "model": model,
                        "provider": res.get("provider", ""),
                        "case_id": case["id"],
                        "category": case["category"],
                        "repeat": rep,
                        "latency_s": res.get("latency_s", ""),
                        "prompt_tokens": res.get("prompt_tokens", ""),
                        "cached_tokens": res.get("cached_tokens", ""),
                        "completion_tokens": res.get("completion_tokens", ""),
                        "reasoning_tokens": res.get("reasoning_tokens", ""),
                        "cost_usd": "" if res.get("cost_usd") is None else res["cost_usd"],
                        "finish_reason": res.get("finish_reason", ""),
                        "reply_chars": len(reply),
                        "figures": " ".join(map(str, check["figures"])),
                        "unlisted_figures": " ".join(map(str, check["unlisted"])),
                        "failed_checks": " | ".join(check["failed"]),
                        "checks_passed": not error and not check["failed"],
                        "error": error,
                        "reply": reply,
                    }
                )
                status = error or ("ok" if not check["failed"] else "check failed")
                print(
                    f"{model:<34} {case['id']:<28} {res.get('latency_s', '-'):>7}s "
                    f"US${float(res.get('cost_usd') or 0):.5f}  {status}"
                )
            if stopped:
                break
        if stopped:
            print(f"spend cap reached (US${spent:.4f} >= US${args.max_cost:g}); stopping")
            break

    if not rows:
        print("nothing ran", file=sys.stderr)
        return 1
    write_csv(out, CSV_FIELDS, rows)
    summary = summarize(rows)
    summary_path = out.with_name(out.stem + "-summary.csv")
    write_csv(summary_path, SUMMARY_FIELDS, summary)
    print()
    print_table(summary, SUMMARY_FIELDS)
    print(f"\nspent US${spent:.4f}. Replies: {out}  Summary: {summary_path}")
    print("Checks are heuristics. Read the 'reply' column before trusting a ranking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
