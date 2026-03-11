from __future__ import annotations


def infer_provider_name(model_name: str | None, provider: str | None = None) -> str:
    explicit = str(provider or "").strip().lower()
    if explicit in {"", "auto"}:
        explicit = ""
    elif explicit != "gemini":
        raise ValueError(
            f"Unsupported provider {provider!r}; this harness supports Gemini models only."
        )

    lowered = str(model_name or "").strip().lower()
    if lowered.startswith("gemini") or "/gemini" in lowered:
        return "gemini"
    raise ValueError(
        f"Unsupported model {model_name!r}; this harness supports Gemini models only."
    )
