import logging
import time
from dataclasses import dataclass

from pydantic_ai import Agent

logger = logging.getLogger(__name__)


@dataclass
class Totals:
    calls: int = 0
    requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    model: str = ""

    def __sub__(self, other: "Totals") -> "Totals":
        """Usage since an earlier snapshot, e.g. what one eval case cost."""
        return Totals(
            self.calls - other.calls,
            self.requests - other.requests,
            self.tool_calls - other.tool_calls,
            self.input_tokens - other.input_tokens,
            self.output_tokens - other.output_tokens,
            self.seconds - other.seconds,
            self.model,
        )


totals = Totals()  # running total for this process; evals read it before and after each case


def describe(exc: BaseException) -> str:
    """One short entry per underlying error. A FallbackModel failure is a group, one error per model."""
    if isinstance(exc, BaseExceptionGroup):
        return " | ".join(describe(e) for e in exc.exceptions)
    model = getattr(exc, "model_name", "")
    status = getattr(exc, "status_code", "")
    return f"{type(exc).__name__} model={model} status={status} {str(exc)[:150]!r}"


def run_tracked(step: str, agent: Agent, prompt: str):
    """Run an agent and log requests, tokens and time. The prompt is never logged (customer text)."""
    began = time.perf_counter()
    try:
        result = agent.run_sync(prompt)
    except Exception as exc:
        logger.warning("%s FAILED after %.1fs: %s", step, time.perf_counter() - began, describe(exc))
        raise
    seconds = time.perf_counter() - began
    usage, model = result.usage, result.response.model_name or ""
    totals.calls += 1
    totals.requests += usage.requests
    totals.tool_calls += usage.tool_calls
    totals.input_tokens += usage.input_tokens
    totals.output_tokens += usage.output_tokens
    totals.seconds += seconds
    totals.model = model
    logger.info(
        "%s model=%s requests=%d tool_calls=%d in=%d out=%d time=%.1fs",
        step, model, usage.requests, usage.tool_calls, usage.input_tokens, usage.output_tokens, seconds,
    )
    return result.output
