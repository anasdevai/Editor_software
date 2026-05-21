"""
Compatibility shim.

The chatbot code is being consolidated under `backend/chatbot/`.
Keep this module so existing imports (`embeddings.embedder`) continue to work.
"""

from chatbot.embeddings.embedder import *  # noqa: F401,F403

