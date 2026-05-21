"""
Compatibility shim.

The chatbot code is being consolidated under `backend/chatbot/`.
Keep this module so existing imports (`chain.rag_chain`) continue to work.
"""

from chatbot.rag.rag_chain import *  # noqa: F401,F403

