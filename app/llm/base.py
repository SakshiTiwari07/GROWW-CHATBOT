"""Errors from the Groq LLM call. The message never includes an API key."""


class LLMError(RuntimeError):
    """The Groq request could not be completed."""
