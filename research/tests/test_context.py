from pathlib import Path
import unittest


class ContextNLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path("models/nlp/english-ewt.udpipe").exists():
            raise unittest.SkipTest("Optional local NLP model not downloaded")
        try:
            from research.supra.nlp_scenes import Extractor
            cls.extractor = Extractor()
        except (ImportError, LookupError) as error:
            raise unittest.SkipTest(f"Optional NLP resources unavailable: {error}")

    def test_preserves_negation_in_source_sentence(self):
        text = "A spacecraft did not launch from the pad."
        result = self.extractor.convert("Launch delayed", text)
        self.assertEqual(result["caption"], text)
        self.assertFalse(result["headline_fallback"])

    def test_preserves_coordinated_event_instead_of_truncating_core(self):
        text = "A Russian cargo ship was hit by a drone and caught fire in the Baltic Sea."
        result = self.extractor.convert("Ship incident", text)
        self.assertIn("was hit", result["caption"])
        self.assertIn("caught fire", result["caption"])

    def test_sentence_splitter_keeps_multiword_name_together(self):
        sentences = self.extractor.parse("Kansas City Current won the championship.")
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0][0], "Kansas City Current won the championship.")

    def test_failed_context_keeps_headline_subject(self):
        result = self.extractor.convert("A newly discovered ancient artifact")
        self.assertTrue(result["headline_fallback"])
        self.assertFalse(result["context_used"])
        self.assertEqual(result["caption"], "A newly discovered ancient artifact.")


class ExtractionTests(unittest.TestCase):
    def test_opening_excludes_browser_boilerplate_and_title(self):
        try:
            from research.supra.context_probe import extract
        except ImportError as error:
            self.skipTest(str(error))
        page = '''<html><head><title>Spacecraft arrives</title>
        <meta name="description" content="A spacecraft has arrived at a rocky planet after a long journey through space."></head>
        <body><article><p>Thank you for visiting this website. You are using a browser version with limited support.</p>
        <h1>Spacecraft arrives</h1>
        <p>A scientific spacecraft has arrived at the planet after a long journey, and its instruments will now study the rocky surface.</p>
        <p>The mission carries cameras and instruments to measure the magnetic field of the planet and examine its surroundings.</p></article></body></html>'''
        summary, opening = extract(page, "Spacecraft arrives", "https://example.com/space")
        self.assertIn("spacecraft", summary)
        self.assertIn("scientific spacecraft", opening)
        self.assertNotIn("browser", opening)
