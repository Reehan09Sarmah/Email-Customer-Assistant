import logging
import os
import sys
import time
from pathlib import Path

import nest_asyncio

sys.path.append(str(Path(__file__).resolve().parents[1]))

nest_asyncio.apply()

from app import db
from app.ai import decide as decide_module
from app.config import settings

question = "I want my money back, this is ridiculous."

START = time.perf_counter()


def say(message: str) -> None:
    print(f"[{time.perf_counter() - START:6.1f}s] {message}", flush=True)


def timed(label: str, func):
    """Wrap a function so it prints before it starts and after it finishes, with the time taken."""

    def wrapper(*args, **kwargs):
        say(f"{label} ...")
        began = time.perf_counter()
        result = func(*args, **kwargs)
        say(f"{label} done in {time.perf_counter() - began:.1f}s")
        return result

    return wrapper


# Show each knowledge base read the agent makes (these lines come from the agent's tools).
class ToolPrinter(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        say(f"Agent tool -> {record.getMessage().removeprefix('Tool call: ')}")


logging.getLogger("app.ai.agent").addHandler(ToolPrinter())
logging.getLogger("app.ai.agent").setLevel(logging.INFO)

# Wrap the steps. decide() looks these names up at call time, so the wrappers take effect.
decide_module.classify = timed(f"Classifying email ({settings.classifier_model})", decide_module.classify)
decide_module.write_reply = timed(f"Agent writing reply ({settings.agent_model})", decide_module.write_reply)
db.get_section = timed("  database: reading section rows", db.get_section)
db.list_sections = timed("  database: listing sections", db.list_sections)

say(f"Reading email: {question!r}")
decision = decide_module.decide(question)
say("Decision made by Python")
print(decision.model_dump_json(indent=2), flush=True)

# Something (likely the Gemini client's background connection) keeps the process alive after
# the work is done. Exit right away instead of waiting for Ctrl+C.
os._exit(0)
