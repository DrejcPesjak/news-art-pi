from pathlib import Path
import math
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
    if generation["backend"] not in ("demo", "supra", "inko"):
        raise ValueError("backend must be demo, supra or inko")
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
    if generation["backend"] in ("supra", "inko"):
        inko = generation["backend"] == "inko"
        if inko and generation["image_size"] not in (256, 384, 512):
            raise ValueError("Inko requires image_size = 256, 384 or 512")
        if not inko and generation["image_size"] != 256:
            raise ValueError("Supra2 currently requires image_size = 256")
        for key, default, minimum, maximum in [
            ("steps", 8 if inko else 25, 1, 100), ("threads", 4, 1, 16),
            ("seed", 0, 0, 2**32-1), ("memory_mib", 480, 128, 2048),
            ("timeout_seconds", 7200 if inko else 900, 30, 10800 if inko else 3600),
        ]:
            value = generation.get(key, default)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{key} must be an integer between {minimum} and {maximum}")
        guidance = generation.get("guidance", 1.0 if inko else 3.0)
        if not isinstance(guidance, (int, float)) or not math.isfinite(guidance) or not 1 <= guidance <= 10:
            raise ValueError("guidance must be finite and between 1 and 10")
    return config

