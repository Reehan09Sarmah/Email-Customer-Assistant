"""Run the golden emails through the pipeline, paced for the Gemini free tier.

    uv run python -m evals.run_evals                       # all cases
    uv run python -m evals.run_evals --cases angry-refund,seam-split
    uv run python -m evals.run_evals --resume              # only cases the last run did not finish
    uv run python -m evals.run_evals --resume --fallback   # same, trying FALLBACK_MODEL on 503/429
"""

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from app.ai import decide as decide_module
from app.ai import usage
from app.ai.agent import support_agent
from app.ai.classifier import classifier
from app.ai.schemas import Decision
from app.config import settings
from evals.golden_emails import GOLDEN_EMAILS

SECONDS_PER_REQUEST = 5  # about 12 requests a minute, under the 15 RPM free-tier limit
RETRY_WAIT_SECONDS = 60
pace = SECONDS_PER_REQUEST  # set to 0 by --fast
LAST_RUN = Path(__file__).with_name(".last_run.json")  # finished results, used by --resume


@dataclass
class SectionMatches(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output.section == ctx.expected_output["section"]


@dataclass
class ActionMatches(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output.action == ctx.expected_output["action"]


@dataclass
class ReplyIsClean(Evaluator):
    """Free check, no API call: reply is not empty, never mentions AI, escalations are signed."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        reply = ctx.output.reply
        if not reply.strip():
            return False
        # "AI" is case-sensitive on purpose: a case-insensitive \bai\b matches French "j'ai".
        if re.search(r"\bAI\b|(?i:artificial intelligence|knowledge base)", reply):
            return False
        return ctx.output.action == "reply" or "Rex and Co." in reply


class QuotaExhausted(Exception):
    pass


names = {case["email"]: case["name"] for case in GOLDEN_EMAILS}
case_usage: dict[str, usage.Totals] = {}
errors: dict[str, str] = {}  # case name -> underlying errors
reuse: dict[str, str] = {}  # email -> Decision JSON loaded from the last run (--resume only)
saved: dict[str, str] = {}  # email -> Decision JSON finished in this run
quota_hit = False


def retryable_status(exc: BaseException) -> int | None:
    """429 or 503 if that is all that went wrong. With --fallback, both models failing arrives as a group."""
    if isinstance(exc, ModelHTTPError):
        statuses = {exc.status_code}
    elif isinstance(exc, BaseExceptionGroup):
        statuses = {retryable_status(e) for e in exc.exceptions}
    else:
        return None
    if statuses and statuses <= {429, 503}:
        return 429 if statuses == {429} else 503
    return None


_classified: dict[str, object] = {}
_real_classify = decide_module.classify


def classify_once(thread: str):
    """A retry after a failed write_reply reuses the classification instead of paying for it again."""
    if thread not in _classified:
        _classified[thread] = _real_classify(thread)
    return _classified[thread]


def decide_with_retry(email: str) -> Decision:
    """Wait a minute and retry on 429 (rate limit) or 503 (overloaded), up to 3 attempts."""
    for attempt in (1, 2, 3):
        try:
            return decide_module.decide(email)
        except Exception as exc:
            status = retryable_status(exc)
            if status is None:
                raise
            if attempt == 3:
                if status == 429:
                    raise QuotaExhausted("Gemini quota exhausted (429). Wait, then run with --resume.") from exc
                raise
            print(f"    HTTP {status}, waiting {RETRY_WAIT_SECONDS}s (attempt {attempt}/3)", flush=True)
            time.sleep(RETRY_WAIT_SECONDS)


def run_pipeline(email: str) -> Decision:
    global quota_hit
    name = names[email]
    if email in reuse:
        print(f"{name}: reused from last run", flush=True)
        saved[email] = reuse[email]
        return Decision.model_validate_json(reuse[email])
    if quota_hit:
        raise QuotaExhausted("not run: quota exhausted earlier in this run")

    before = replace(usage.totals)
    try:
        decision = decide_with_retry(email)
    except Exception as exc:
        errors[name] = usage.describe(exc)  # shown in the summary, which the log lines scroll past
        quota_hit = quota_hit or isinstance(exc, QuotaExhausted)
        raise
    used = usage.totals - before
    case_usage[name] = used
    saved[email] = decision.model_dump_json()
    LAST_RUN.write_text(json.dumps(saved), encoding="utf-8")

    pause = pace * used.requests
    print(f"{name}: {used.requests} requests, {used.seconds:.1f}s, pausing {pause}s", flush=True)
    time.sleep(pause)  # pace by what the case really cost
    return decision


def make_dataset(cases: list[dict]) -> Dataset:
    return Dataset(
        name="golden-emails",
        cases=[
            Case(
                name=case["name"],
                inputs=case["email"],
                expected_output={"section": case["section"], "action": case["action"]},
            )
            for case in cases
        ],
        evaluators=[SectionMatches(), ActionMatches(), ReplyIsClean()],
    )


dataset = make_dataset(GOLDEN_EMAILS)


def print_summary(report) -> None:
    wrong = [
        f"  {case.name}: failed {', '.join(n for n, r in case.assertions.items() if not r.value)}"
        for case in report.cases
        if not all(r.value for r in case.assertions.values())
    ]
    not_run = [f"  {f.name}: {errors.get(f.name, f.error_message)[:600]}" for f in report.failures]

    print("\n=== Summary ===")
    print(f"passed {len(report.cases) - len(wrong)}/{len(report.cases) + len(report.failures)}")
    if wrong:
        print("wrong answers:\n" + "\n".join(wrong))
    if not_run:
        print("errors / not run (not wrong answers):\n" + "\n".join(not_run))

    print("\n=== Usage per case ===")
    print(f"{'case':32} {'requests':>8} {'tool':>5} {'in tok':>8} {'out tok':>8} {'secs':>6}  model")
    for name, u in case_usage.items():
        print(f"{name:32} {u.requests:8} {u.tool_calls:5} {u.input_tokens:8} {u.output_tokens:8} {u.seconds:6.1f}  {u.model}")
    all_used = list(case_usage.values())
    print(
        f"{'TOTAL':32} {sum(u.requests for u in all_used):8} {sum(u.tool_calls for u in all_used):5} "
        f"{sum(u.input_tokens for u in all_used):8} {sum(u.output_tokens for u in all_used):8} "
        f"{sum(u.seconds for u in all_used):6.1f}"
    )


def main() -> None:
    global pace
    parser = argparse.ArgumentParser(description="Run the golden email evals.")
    parser.add_argument("--cases", help="comma-separated case names (default: all)")
    parser.add_argument("--resume", action="store_true", help="reuse finished results from the last run")
    parser.add_argument("--fallback", action="store_true", help="on a model error, try FALLBACK_MODEL (off by default)")
    parser.add_argument(
        "--fast", action="store_true",
        help="paid tier: no pacing, 4 cases at once (the per-case usage table is then approximate)",
    )
    args = parser.parse_args()
    if args.fast:
        pace = 0

    cases = GOLDEN_EMAILS
    if args.cases:
        wanted = [name.strip() for name in args.cases.split(",")]
        unknown = set(wanted) - set(names.values())
        if unknown:
            parser.error(f"unknown case(s): {', '.join(sorted(unknown))}")
        cases = [case for case in GOLDEN_EMAILS if case["name"] in wanted]
    if args.resume and LAST_RUN.exists():
        reuse.update(json.loads(LAST_RUN.read_text(encoding="utf-8")))
        saved.update(reuse)  # keep results of cases not reached yet when the file is rewritten

    decide_module.classify = classify_once
    if not args.fallback:  # scored runs use one model; the agents default to the fallback chain
        classifier.model = settings.classifier_model
        support_agent.model = settings.agent_model

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("app").setLevel(logging.INFO)

    # One case at a time keeps the free tier under its per-minute limit; --fast is for paid tiers.
    concurrency = 4 if args.fast else 1
    report = make_dataset(cases).evaluate_sync(run_pipeline, max_concurrency=concurrency, progress=False)
    report.print(include_input=False, include_output=True, include_durations=False)
    print_summary(report)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # A worker thread may be sleeping or waiting on Gemini; exit now instead of hanging.
        print("\nInterrupted. Finished cases are saved; continue with --resume.", flush=True)
        os._exit(130)
