"""NLP/profile detection services."""

from .layer import run_nlp_layer
from .llm_action_context import build_action_llm_context_block

__all__ = ["run_nlp_layer", "build_action_llm_context_block"]

