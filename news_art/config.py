from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Ubuntu 22.04 / Python 3.10 development laptops
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent


def load(path=ROOT / "config.toml"):
    with open(path, "rb") as stream:
        config = tomllib.load(stream)
    news, generation, display = (config[key] for key in ("news", "generation", "display"))
    if not 30 <= display["refresh_seconds"] <= 86400:
        raise ValueError("refresh_seconds must be between 30 and 86400")
    if generation["backend"] != "demo":
        raise ValueError("Only the demo backend is available in this release")
    if not 64 <= generation["image_size"] <= 1024:
        raise ValueError("image_size must be between 64 and 1024")
    if not 1 <= news["max_age_hours"] <= 168:
        raise ValueError("max_age_hours must be between 1 and 168")
    if len(news["country"]) != 2 or ":" in news["language"]:
        raise ValueError("Use a two-letter country and a language such as en-US")
    for key in ("query", "language", "country"):
        if not isinstance(news[key], str):
            raise ValueError(f"news.{key} must be text")
    if not isinstance(generation["style"], str) or not isinstance(display["title"], str):
        raise ValueError("style and title must be text")
    return config

