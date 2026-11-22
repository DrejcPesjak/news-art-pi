import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from news_art.article import context_for, extract
from news_art.captions import prepare
from news_art.generator import prompt_for


class CaptionTests(unittest.TestCase):
    def test_caption_child_receives_story_as_data(self):
        story = {"headline": "A ship; $(touch should-never-execute)", "context": "A ship at sea."}
        report = {"scene": "a ship at sea", "article": {"status": "available"}}
        with patch("news_art.captions.subprocess.run", return_value=SimpleNamespace(stdout=json.dumps(report))) as run:
            self.assertEqual(prepare(story, True), report)
        args, kwargs = run.call_args
        self.assertIn("--fetch-context", args[0])
        self.assertNotIn(story["headline"], args[0])
        self.assertEqual(json.loads(kwargs["input"]), story)
        self.assertTrue(kwargs["check"])
        self.assertEqual(kwargs["timeout"], 120)

    def test_style_and_caption_diagnostics_saved_with_story(self):
        story = {"headline": "Names that should not enter the prompt"}
        report = {"scene": "a couple getting married", "converter": "brief-v3", "article": {"status": "available"}}
        with patch("news_art.generator.prepare", return_value=report):
            prompt = prompt_for(story, "oil painting", fetch_context=True)
        self.assertEqual(prompt, "a couple getting married. oil painting")
        self.assertEqual(story["caption"]["prompt"], prompt)
        self.assertEqual(story["caption"]["converter"], "brief-v3")

    def test_publisher_context_avoids_navigation_and_boilerplate(self):
        body = b'''<html><head><meta name="description" content="A spacecraft has arrived at a planet after a long journey through space."></head>
        <nav><p>This navigation paragraph must not be used as article context for the image.</p></nav>
        <article><p>Subscribe to read the latest stories and receive updates straight to your inbox.</p>
        <p>A spacecraft has reached the planet and will begin studying its surface using cameras and scientific instruments.</p></article></html>'''
        summary, opening = extract(body, "Spacecraft arrives")
        self.assertIn("spacecraft", summary)
        self.assertIn("scientific instruments", opening)
        self.assertNotIn("navigation", opening)
        self.assertNotIn("Subscribe", opening)

    def test_failed_context_fetch_is_explicit_headline_fallback(self):
        with patch("news_art.article.resolve", side_effect=OSError("Offline")):
            report = context_for({"headline": "A ship", "url": "https://news.google.com/example"})
        self.assertEqual(report["context"], "")
        self.assertEqual(report["status"], "failed")
        self.assertIn("Offline", report["error"])


if __name__ == "__main__":
    unittest.main()
