"""Shared model configuration for agents-kg.

All Gemini model names used across the pipeline are centralised here
so they can be updated in one place.
"""

MODEL_EMBEDDING = "gemini-embedding-2"
MODEL_EMBEDDING_OLD = "gemini-embedding-2-preview"  # kept for migration
MODEL_EXTRACT = "gemini-3.5-flash-lite"
MODEL_SYNTHESIS = "gemini-3.6-flash"
