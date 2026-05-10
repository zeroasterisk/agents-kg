import os
import sys
import pytest
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_has_genai_creds = bool(os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))

@pytest.mark.skipif(not _has_genai_creds, reason="No Gemini/Vertex AI credentials configured")
def test_genai_access():
    kwargs_gen = {}
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        kwargs_gen["enterprise"] = True
        kwargs_gen["project"] = os.environ["GOOGLE_CLOUD_PROJECT"]
        kwargs_gen["location"] = "global"
    
    kwargs_embed = {}
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        kwargs_embed["enterprise"] = True
        kwargs_embed["project"] = os.environ["GOOGLE_CLOUD_PROJECT"]
        kwargs_embed["location"] = "us-central1"
    
    print(f"Initializing generation client with kwargs: {kwargs_gen}")
    client_gen = genai.Client(**kwargs_gen)
    
    print(f"Initializing embedding client with kwargs: {kwargs_embed}")
    client_embed = genai.Client(**kwargs_embed)
    
    # Test generation
    model_name = "gemini-3.5-flash-lite"
    print(f"Testing generation with model: {model_name} at location: global")
    
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="medium"
        )
    )
    
    try:
        response = client_gen.models.generate_content(
            model=model_name,
            contents="Hello, are you working?",
            config=config
        )
        assert response.text, "Generation returned empty text"
    except Exception as e:
        pytest.fail(f"Generation FAILED: {e}")

    # Test embedding
    embed_model = "gemini-embedding-2"
    try:
        result = client_embed.models.embed_content(
            model=embed_model,
            contents="Hello world",
        )
        assert len(result.embeddings) > 0, "Embedding returned no results"
    except Exception as e:
        pytest.fail(f"Embedding FAILED: {e}")
