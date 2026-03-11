"""Smoke test that verifies Gemini API connectivity without running inference.

Run with:
    pytest tests/test_smoke_api.py -v

Skipped automatically when no API key is configured.
"""

from __future__ import annotations

import pytest

from debate_v_majority.engines.gemini_api import _resolve_gemini_api_key
from debate_v_majority.engines.engine_impl import GEMINI_3_FLASH_MODEL

_api_key = _resolve_gemini_api_key(None)
_skip_reason = "GEMINI_API_KEY not set (check .env or environment)"


@pytest.mark.skipif(_api_key is None, reason=_skip_reason)
def test_gemini_api_key_is_valid():
    """Lists models to confirm the API key is accepted."""
    from google import genai

    client = genai.Client(api_key=_api_key)
    available = {m.name for m in client.models.list()}

    assert len(available) > 0, "API returned zero models — key may be invalid"


@pytest.mark.skipif(_api_key is None, reason=_skip_reason)
def test_gemini_3_flash_model_available():
    """Checks that a gemini-3-flash-preview variant is listed in the API."""
    from google import genai

    client = genai.Client(api_key=_api_key)
    available = sorted(m.name for m in client.models.list())

    target = f"models/{GEMINI_3_FLASH_MODEL}"
    flash_variants = [n for n in available if GEMINI_3_FLASH_MODEL in n]

    assert flash_variants, (
        f"No {GEMINI_3_FLASH_MODEL} variant found in API. "
        f"Available gemini models: {[n for n in available if 'gemini' in n]}"
    )
    if target not in available:
        print(
            f"\nNote: exact name '{target}' not in API, "
            f"but these variants exist: {flash_variants}"
        )
