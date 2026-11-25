"""Semantic regression checks for the optional laptop caption prototype."""
import unittest
from pathlib import Path

from research.supra.brief_scenes import Pruner


class BriefSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.pruner = Pruner(Path.home()/".cache/news-art-nltk")
        except (ImportError, LookupError) as error:
            raise unittest.SkipTest(f"Optional NLP resources unavailable: {error}")

    def test_strips_person_and_geographic_names_but_keeps_role(self):
        result = self.pruner.convert(
            "British swimmer Lewis Pugh swims in the Strait of Hormuz")
        self.assertIn("swimmer", result["scene"])
        self.assertNotIn("Lewis", result["scene"])
        self.assertNotIn("Hormuz", result["scene"])
        self.assertIn("sea", result["scene"])

    def test_couple_does_not_imply_marriage(self):
        result = self.pruner.convert("A couple walks near a river")
        self.assertIn("couple", result["scene"])
        self.assertNotIn("married", result["scene"])

    def test_context_supplies_role_missing_from_person_names(self):
        result = self.pruner.convert(
            "Miley Cyrus and Maxx Morando Are Married",
            "The couple tied the knot in Palm Springs.")
        self.assertEqual(result["prompt"], "a couple getting married. oil painting")


if __name__ == "__main__":
    unittest.main()
