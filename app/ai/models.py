from pydantic_ai.models.fallback import FallbackModel

from app.config import settings


def build_model(primary: str):
    """The model an agent uses: the primary, then settings.fallback_model if the primary errors."""
    if not settings.fallback_model:
        return primary
    return FallbackModel(primary, settings.fallback_model)  # falls back on any ModelAPIError (503, 429, ...)
