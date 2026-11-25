import unittest

from news_art.scenes import scene_for


class SceneTests(unittest.TestCase):
    def test_specific_event_precedes_broad_topic(self):
        examples = [
            ("Cargo ship ablaze after drone strike", "ship-fire"),
            ("New pancreatic cancer drug stops tumor growth: Study", "medical-research"),
            ("Drug lord on wanted list arrested", "crime"),
            ("Trump vows to cap credit card interest rates", "finance"),
            ("UCLA QB injured in loss to Michigan", "american-football"),
        ]
        for headline, expected in examples:
            with self.subTest(headline=headline):
                self.assertEqual(scene_for(headline)[0], expected)

    def test_word_boundaries_avoid_accidental_topics(self):
        for headline in ["Spain opens a new museum", "Chair made from recycled wood", "Mercury levels measured in river water"]:
            with self.subTest(headline=headline):
                self.assertEqual(scene_for(headline)[0], "headline-fallback")

    def test_known_topic_removes_names_and_numbers_from_visual_prompt(self):
        prompt = scene_for("Kansas City Current capture NWSL trophy in 2-1 win")[1] + " In the style of Picasso"
        self.assertIn("soccer", prompt)
        self.assertIn("trophy", prompt)
        self.assertIn("Women", prompt)
        self.assertIn("Picasso", prompt)
        for unwanted in ["Kansas", "2-1", "headline", "lettering"]:
            self.assertNotIn(unwanted, prompt)

    def test_unknown_subject_is_preserved_and_boilerplate_removed(self):
        self.assertEqual(scene_for("Exclusive | A new museum opens — live reaction"),
                         ("headline-fallback", "A new museum opens."))

    def test_space_scene_and_arbitrary_style(self):
        prompt = scene_for("BepiColombo reaches Mercury")[1] + " Watercolor painting."
        self.assertIn("spacecraft", prompt)
        self.assertTrue(prompt.endswith("Watercolor painting."))
