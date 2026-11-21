from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from news_art.config import load
from news_art.generator import generate, prompt_for
from news_art.news import feed_url, parse
from news_art.worker import cycle, read_state

NOW = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)
STORY = {"id": "story1", "headline": "A test headline", "source": "Example News",
         "published": NOW.isoformat(), "url": "https://example.com/story"}


def demo_config():
    config = load()
    config["generation"]["backend"] = "demo"
    return config


class PipelineTests(unittest.TestCase):
    def test_feed_filter_is_one_encoded_google_feed(self):
        news = load()["news"]
        self.assertNotIn("/search?", feed_url(news))
        news["query"] = "sports Spain"
        self.assertIn("/search?", feed_url(news))
        self.assertIn("q=sports+Spain", feed_url(news))

    def test_parser_skips_stale_bad_dates_and_duplicate_stories(self):
        item = '''<item><title>A &amp; B - Example</title><source>Example</source>
        <link>https://example.com/article</link><guid>one</guid>
        <pubDate>Sat, 21 Nov 2026 11:00:00 GMT</pubDate></item>'''
        stale = item.replace("21 Nov", "01 Nov").replace("one", "old")
        invalid = item.replace("Sat, 21 Nov 2026 11:00:00 GMT", "invalid")
        stories = parse(f"<rss><channel>{item}{item}{stale}{invalid}</channel></rss>", now=NOW)
        self.assertEqual(len(stories), 1)
        self.assertEqual(stories[0]["headline"], "A & B")

    def test_all_three_demo_images_are_square_and_distinct(self):
        images = [generate(STORY, demo_config()["generation"], i)[0] for i in range(4)]
        self.assertTrue(all(image.size == (256, 256) and image.mode == "RGB" for image in images))
        self.assertEqual(len({image.tobytes() for image in images[:3]}), 3)
        self.assertEqual(images[0].tobytes(), images[3].tobytes())

    def test_style_is_in_prompt(self):
        self.assertIn("Picasso", prompt_for(STORY, "In the style of Picasso"))

    def test_history_rotation_persists_and_failure_keeps_previous_poster(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, config = Path(temp), demo_config()
            first = cycle(config, directory, [STORY])
            second_story = dict(STORY, id="story2", headline="Another headline")
            second = cycle(config, directory, [STORY, second_story])
            self.assertEqual(second["story"]["id"], "story2")
            self.assertNotEqual(first["image"], second["image"])
            with Image.open(directory / second["image"]) as image:
                self.assertEqual(image.size, (256, 256))
            before = (directory / "state.json").read_bytes()
            with patch("news_art.worker.fetch", side_effect=OSError("Offline")):
                with self.assertRaises(OSError):
                    cycle(config, directory)
            self.assertEqual((directory / "state.json").read_bytes(), before)
            self.assertEqual(read_state(directory)["sequence"], 2)

    def test_invalid_config_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path("config.toml").read_text().replace("refresh_seconds = 600", "refresh_seconds = 0")
            path = Path(temp) / "bad.toml"
            path.write_text(source)
            with self.assertRaises(ValueError):
                load(path)


if __name__ == "__main__":
    unittest.main()

