import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from news_art.config import load
from news_art.inko import generate


class InkoIntegrationTests(unittest.TestCase):
    def test_candidate_resolution_and_limits(self):
        source = Path("config.inko.toml").read_text()
        config = load(Path("config.inko.toml"))
        self.assertEqual(config["generation"]["backend"], "inko")
        self.assertEqual(config["generation"]["image_size"], 384)
        for old, new in [("image_size = 384", "image_size = 448"),
                         ("timeout_seconds = 7200", "timeout_seconds = 0"),
                         ("guidance = 1.0", "guidance = nan")]:
            with self.subTest(new=new), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "config.toml"
                path.write_text(source.replace(old, new))
                with self.assertRaises(ValueError):
                    load(path)

    def test_three_isolated_stages_share_one_timeout_and_record_metrics(self):
        config = load(Path("config.inko.toml"))["generation"]
        prompt = "A ship; $(touch should-not-run). oil painting"
        observed = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "bundle/models"
            models.mkdir(parents=True)
            (models / "recipe.json").write_text(json.dumps(dict(variant="q4-distilled-3000-gptq")))
            state = root / "state"
            state.mkdir()
            def run(command, timeout, environment):
                stage = command[command.index("research.edge_quality.staged")+1]
                work = Path(command[command.index("--work")+1])
                observed.append((stage, command, timeout, environment))
                (work / f"{stage}.json").write_text(json.dumps(dict(
                    forbidden_imports_loaded=[], peak_rss_mib=10, allocation_limit_mib=480)))
                if stage == "decode":
                    Image.new("RGB", (384, 384), "blue").save(work / "image.png")
            with patch.dict(os.environ, NEWS_ART_INKO_DIR=str(root / "bundle")), \
                 patch("news_art.inko.run_pipeline", run), \
                 patch("news_art.inko.time.monotonic", side_effect=[100, 101, 104, 109, 110]):
                image = generate(prompt, config, 9, state)
            self.assertEqual([r[0] for r in observed], ["encode", "sample", "decode"])
            self.assertEqual([r[2] for r in observed], [7199, 7196, 7191])
            for _, command, _, environment in observed:
                self.assertEqual(command[command.index("--prompt")+1], prompt)
                self.assertEqual(command[command.index("--seed")+1], "9")
                self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(image.size, (384, 384))
            self.assertEqual(set(image.info["inference"]["stages"]), {"encode", "sample", "decode"})
            self.assertEqual(list((state / "generation").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
