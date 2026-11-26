from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
from news_art.config import load
from news_art.generator import generate
from news_art.supra import run_pipeline
from news_art.worker import cycle, read_state, refresh_delay

STORY = {"id": "news-test", "headline": "A rocket launches from Spain",
         "source": "Test", "url": "https://example.com/news",
         "published": datetime.now(timezone.utc).isoformat()}


class SupraIntegrationTests(unittest.TestCase):
    def test_supra_config_rejects_unsupported_resolution_and_bad_parameters(self):
        original = Path("config.supra.toml").read_text()
        for old, new in [("image_size = 256", "image_size = 512"),
                         ("steps = 25", "steps = 0"),
                         ("guidance = 3.0", "guidance = nan")]:
            with self.subTest(new=new), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)/"config.toml"
                path.write_text(original.replace(old, new))
                with self.assertRaises(ValueError): load(path)

    def test_adapter_passes_prompt_as_one_argument_and_validates_image(self):
        config = load(Path("config.supra.toml"))["generation"]
        story = dict(STORY, headline="An unusual event; $(touch should-never-execute)")
        observed = []
        def pipeline(command, timeout, environment):
            observed.append(command)
            output = Path(command[command.index("--output")+1])
            Image.new("RGB", (256,256), "blue").save(output/"image.png")
        caption = {"scene": story["headline"], "article": {"status": "failed"}}
        with tempfile.TemporaryDirectory() as temporary, patch("news_art.supra.run_pipeline", pipeline), patch("news_art.generator.prepare", return_value=caption):
            image, prompt = generate(story, config, 7, Path(temporary))
            self.assertEqual(image.size, (256,256))
            command = observed[0]
            self.assertEqual(command[command.index("--prompt")+1], prompt)
            self.assertIn(story["headline"], prompt)
            self.assertEqual(command[command.index("--seed")+1], "7")
            self.assertEqual(list((Path(temporary)/"generation").iterdir()), [])

    def test_model_failure_keeps_previous_state_and_records_error(self):
        config = load(Path("config.supra.toml"))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch("news_art.worker.generate", return_value=(Image.new("RGB",(256,256)), "prompt")):
                cycle(config, directory, [STORY])
            before = (directory/"state.json").read_bytes()
            with patch("news_art.worker.generate", side_effect=RuntimeError("Inference failed")):
                with self.assertRaises(RuntimeError): cycle(config, directory, [STORY])
            self.assertEqual((directory/"state.json").read_bytes(), before)
            self.assertEqual(read_state(directory)["backend"], "supra")
            self.assertIn("failed", (directory/"job.json").read_text())

    def test_schedule_counts_generation_time(self):
        self.assertEqual(refresh_delay(600, 570), 30)
        self.assertEqual(refresh_delay(600, 620), 1)

    def test_timeout_terminates_descendants(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary)/"orphan.txt"
            child = "import time,pathlib; time.sleep(1); pathlib.Path("+repr(str(marker))+").write_text('orphan')"
            supervisor = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"+repr(child)+"]); time.sleep(10)"
            with self.assertRaises(subprocess.TimeoutExpired):
                run_pipeline([sys.executable,"-c",supervisor], .3, os.environ.copy())
            time.sleep(1.1)
            self.assertFalse(marker.exists())


if __name__ == "__main__": unittest.main()
