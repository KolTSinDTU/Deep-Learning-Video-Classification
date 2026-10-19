"""Check dataset contracts, validation metrics, and checkpoint selection."""

import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from PIL import Image
import torch
from torch import nn

from video_classification.data import build_dataloaders
from video_classification.models import build_model
from video_classification.train import _validate, train


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.output_root = Path(__file__).resolve().parents[1] / "outputs"
        self.root = self.output_root / f"_test_pipeline_{uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(self.remove_fixture)
        self.data = self.root / "data"
        (self.data / "metadata").mkdir(parents=True)
        self.write_split("train")
        self.write_split("val")
        # Deliberately no test split: training must not need or read it.
        self.config = {
            "experiment": {"name": "test", "seed": 42, "output_dir": str(self.root / "outputs")},
            "dataset": {"root": str(self.data), "num_classes": 2,
                        "num_frames": 2, "image_size": 16, "regime": "original"},
            "model": {"name": "per_frame"},
            "training": {"epochs": 3, "batch_size": 3, "validation_batch_size": 1,
                         "num_workers": 0, "learning_rate": 0.001, "device": "cpu"},
        }

    def remove_fixture(self):
        # Verify the exact workspace fixture before recursively removing it.
        resolved = self.root.resolve()
        if resolved.parent != self.output_root.resolve() or not resolved.name.startswith("_test_pipeline_"):
            raise RuntimeError(f"Refusing to remove unexpected fixture path: {resolved}")
        shutil.rmtree(resolved)

    def write_split(self, split, *, overlap=False, swap_labels=False):
        rows = []
        for label, action in enumerate(("squats", "jumps")):
            video_id = f"{'train' if overlap else split}_{action}"
            folder = self.data / "frames" / split / action / video_id
            folder.mkdir(parents=True, exist_ok=True)
            for index in (1, 2):
                Image.new("RGB", (20, 18), (label * 80 + index * 25, 20, 40)).save(
                    folder / f"frame_{index}.jpg"
                )
            rows.append({
                "video_name": video_id, "video_path": f"{action}/{video_id}.avi",
                "action": action, "label": 1 - label if swap_labels else label,
            })
        with (self.data / "metadata" / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_loaders_preserve_labels_frame_order_and_validation_shape(self):
        loaders = build_dataloaders(self.config)
        training = loaders["train"].dataset
        self.assertEqual(len(training), 4)
        self.assertEqual([training[i][1] for i in range(4)], [0, 0, 1, 1])
        self.assertEqual(training.class_to_label, {"squats": 0, "jumps": 1})
        clips, labels = next(iter(loaders["val"]))
        self.assertEqual(clips.shape, (1, 3, 2, 16, 16))
        self.assertEqual(labels.tolist(), [0])
        self.assertLess(clips[0, 0, 0].mean(), clips[0, 0, 1].mean())
        self.assertTrue(((clips >= 0) & (clips <= 1)).all())

    def test_rejects_inconsistent_validation_labels(self):
        self.write_split("val", swap_labels=True)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            build_dataloaders(self.config)

    def test_rejects_exact_video_overlap(self):
        self.write_split("val", overlap=True)
        with self.assertRaisesRegex(ValueError, "both train and validation"):
            build_dataloaders(self.config)

    def test_missing_frame_is_reported_before_training(self):
        frame = self.data / "frames/train/squats/train_squats/frame_2.jpg"
        frame.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "frame_2.jpg"):
            build_dataloaders(self.config)

    def test_validation_uses_video_nll_and_weights_partial_batches(self):
        class KnownProbabilities(nn.Module):
            def predict_video(self, values):
                assert not self.training
                assert not torch.is_grad_enabled()
                return values

        batches = [
            (torch.tensor([[0.9, 0.1], [0.2, 0.8]]), torch.tensor([0, 0])),
            (torch.tensor([[0.3, 0.7]]), torch.tensor([1])),
        ]
        metrics = _validate(KnownProbabilities(), batches, torch.device("cpu"))
        expected_loss = -torch.tensor([0.9, 0.2, 0.7]).log().mean().item()
        self.assertAlmostEqual(metrics["loss"], expected_loss, places=6)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["count"], 3)

    def test_saves_best_validation_checkpoint_and_protects_previous_runs(self):
        validation = [
            {"loss": 0.7, "accuracy": 0.5, "count": 2},
            {"loss": 0.6, "accuracy": 0.5, "count": 2},
            {"loss": 0.5, "accuracy": 0.25, "count": 2},
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            with patch("video_classification.train._validate", side_effect=validation):
                run = train(self.config)
            original_checkpoint = (run / "best.pt").read_bytes()
            # A real validation pass in a second, clearly marked smoke run.
            smoke = train(self.config, smoke_test=True)
        self.assertNotEqual(run, smoke)
        self.assertTrue(smoke.name.startswith("smoke_"))
        self.assertEqual((run / "best.pt").read_bytes(), original_checkpoint)
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["class_to_label"], {"squats": 0, "jumps": 1})
        self.assertFalse(checkpoint["smoke_test"])
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        with (run / "history.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        self.assertEqual(len(history), 3)
        self.assertEqual([int(row["train_frames"]) for row in history], [4, 4, 4])
        summary = json.loads((run / "summary.json").read_text())
        self.assertEqual(summary["best_epoch"], 2)
        smoke_summary = json.loads((smoke / "summary.json").read_text())
        self.assertTrue(smoke_summary["smoke_test"])
        self.assertEqual(smoke_summary["completed_epochs"], 1)
        # Training must not mutate the caller's config.
        self.assertEqual(self.config["training"]["epochs"], 3)
        self.assertEqual(self.config["experiment"]["output_dir"], str(self.root / "outputs"))


if __name__ == "__main__":
    unittest.main()
