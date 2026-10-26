"""Check reproducible sampling and native video extraction without real data."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from video_classification.prepare_frames import extract_video, sample_indices


class PrepareFramesTests(unittest.TestCase):
    def test_sampling_includes_endpoints_without_duplicates(self):
        self.assertEqual(sample_indices(613, 10), [0, 68, 136, 204, 272, 340, 408, 476, 544, 612])
        self.assertEqual(sample_indices(3, 3), [0, 1, 2])
        self.assertEqual(sample_indices(8, 1), [4])
        with self.assertRaises(ValueError):
            sample_indices(2, 10)

    def test_extraction_uses_decoded_count_saves_indices_and_reuses_cache(self):
        output_root = Path(__file__).resolve().parents[1] / "outputs"
        output_root.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix="_test_extraction_", dir=output_root) as temporary:
            root = Path(temporary)
            source, destination = root / "source.avi", root / "cache"
            source.write_bytes(b"fixture")

            def fake_run(arguments):
                if arguments[0] == "ffprobe":
                    self.assertIn("-count_frames", arguments)
                    # A misleading container count must not select index 613.
                    return json.dumps({"streams": [{"width": 320, "height": 240,
                                                    "nb_frames": "614", "nb_read_frames": "613"}]})
                selection = arguments[arguments.index("-vf") + 1]
                self.assertIn("eq(n\\,612)", selection)
                self.assertNotIn("scale=", selection)
                pattern = arguments[-1]
                for index in range(1, 11):
                    Image.new("RGB", (320, 240)).save(pattern.replace("%d", str(index)))
                return ""

            with patch("video_classification.prepare_frames._run", side_effect=fake_run) as run:
                self.assertEqual(extract_video(source, destination, 10), "created")
                self.assertEqual(run.call_count, 2)
                manifest = json.loads((destination / "sampling.json").read_text())
                self.assertEqual(manifest["frame_indices_zero_based"], sample_indices(613, 10))
                self.assertEqual(extract_video(source, destination, 10), "reused")
                self.assertEqual(run.call_count, 2)
                with self.assertRaisesRegex(ValueError, "Cache does not match"):
                    extract_video(source, destination, 5)


if __name__ == "__main__":
    unittest.main()
