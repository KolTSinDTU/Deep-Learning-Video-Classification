"""Check dataset contracts, validation metrics, and checkpoint selection."""

import contextlib
import copy
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
from video_classification.config import load_config
from video_classification.initialization import initialize_model
from video_classification.models import build_model
from video_classification.train import _validate, main, train


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

    def test_video_models_load_whole_videos_for_training(self):
        for name in ("late_fusion", "early_fusion", "cnn3d"):
            with self.subTest(model=name):
                self.config["model"]["name"] = name
                loaders = build_dataloaders(self.config)
                training = loaders["train"].dataset
                self.assertFalse(training.per_frame)
                self.assertEqual(len(training), 2)
                clips, labels = next(iter(loaders["train"]))
                self.assertEqual(clips.shape, (2, 3, 2, 16, 16))
                self.assertEqual(sorted(labels.tolist()), [0, 1])
                self.assertFalse(loaders["val"].dataset.per_frame)

    def test_video_training_saves_video_metrics_and_reloadable_checkpoint(self):
        for name, description in (("late_fusion", "Mean of frame features"),
                                  ("early_fusion", "Stack RGB frames"),
                                  ("cnn3d", "Spatiotemporal 3D convolutions")):
            with self.subTest(model=name):
                self.check_video_training_run(name, description)

    def check_video_training_run(self, name, aggregation_description):
        self.config["model"]["name"] = name
        with contextlib.redirect_stdout(io.StringIO()):
            run = train(self.config, smoke_test=True)
        with (run / "history.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        self.assertEqual(len(history), 1)
        self.assertEqual(int(history[0]["train_videos"]), 2)
        self.assertEqual(int(history[0]["val_videos"]), 2)
        self.assertIn("train_video_loss", history[0])
        self.assertIn("train_video_accuracy", history[0])
        self.assertNotIn("train_frame_loss", history[0])
        environment = json.loads((run / "environment.json").read_text())
        self.assertEqual(environment["training_unit"], "video")
        self.assertIn(aggregation_description, environment["video_aggregation"])
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["config"]["model"]["name"], name)
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        validation = build_dataloaders(self.config)["val"]
        metrics = _validate(model, validation, torch.device("cpu"), video_logits=True)
        self.assertAlmostEqual(metrics["loss"], checkpoint["validation"]["loss"], places=6)
        self.assertEqual(metrics["accuracy"], checkpoint["validation"]["accuracy"])

    def test_groupnorm_video_checkpoints_reload_with_the_saved_architecture(self):
        self.config["model"]["normalization"] = "group"
        for name in ("late_fusion", "early_fusion", "cnn3d"):
            with self.subTest(model=name):
                self.check_video_training_run(name, "")

    def source_checkpoint(self):
        model = build_model(self.config)
        return {
            "config": copy.deepcopy(self.config),
            "class_to_label": {"squats": 0, "jumps": 1},
            "model_state_dict": model.state_dict(),
            "epoch": 7,
            "smoke_test": False,
            # Deliberately not loadable as optimizer state: initialization must ignore it.
            "optimizer_state_dict": {"sentinel": "do not reuse"},
        }

    def test_warm_start_loads_exact_weights_and_records_source(self):
        checkpoint = self.source_checkpoint()
        source = self.root / "source.pt"
        torch.save(checkpoint, source)
        self.config["model"]["name"] = "late_fusion"
        self.config["training"]["init_checkpoint"] = str(source)
        model = build_model(self.config)
        provenance = initialize_model(model, self.config, checkpoint["class_to_label"])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, checkpoint["model_state_dict"][name])
        self.assertEqual(provenance["source_epoch"], 7)
        self.assertEqual(provenance["source_model"], "per_frame")
        self.assertFalse(provenance["optimizer_state_reused"])
        self.assertEqual(len(provenance["source_sha256"]), 64)

    def test_warm_start_training_uses_fresh_optimizer_and_preserves_source(self):
        source = self.root / "source.pt"
        torch.save(self.source_checkpoint(), source)
        before = source.read_bytes()
        self.config["model"]["name"] = "late_fusion"
        self.config["training"]["init_checkpoint"] = str(source)
        self.config["training"]["learning_rate"] = 0.0001
        with contextlib.redirect_stdout(io.StringIO()):
            run = train(self.config, smoke_test=True)
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["initialization"]["method"], "per_frame_checkpoint")
        self.assertEqual(checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"], 0.0001)
        self.assertEqual(checkpoint["config"]["training"]["init_checkpoint"], str(source.resolve()))
        self.assertEqual(source.read_bytes(), before)
        rebuilt = build_model(checkpoint["config"])
        rebuilt.load_state_dict(checkpoint["model_state_dict"], strict=True)

    def test_warm_start_rejects_incompatible_sources(self):
        original = self.source_checkpoint()
        source = self.root / "source.pt"
        self.config["model"]["name"] = "late_fusion"
        self.config["training"]["init_checkpoint"] = str(source)
        for problem in ("model", "smoke", "labels", "regime", "normalization", "shape"):
            with self.subTest(problem=problem):
                checkpoint = copy.deepcopy(original)
                if problem == "model":
                    checkpoint["config"]["model"]["name"] = "cnn3d"
                elif problem == "smoke":
                    checkpoint["smoke_test"] = True
                elif problem == "labels":
                    checkpoint["class_to_label"] = {"squats": 1, "jumps": 0}
                elif problem == "regime":
                    checkpoint["config"]["dataset"]["regime"] = "no_leakage"
                elif problem == "normalization":
                    checkpoint["config"]["model"]["normalization"] = "group"
                elif problem == "shape":
                    checkpoint["model_state_dict"]["classifier.weight"] = torch.zeros(3, 128)
                torch.save(checkpoint, source)
                with self.assertRaises(ValueError):
                    initialize_model(build_model(self.config), self.config,
                                     {"squats": 0, "jumps": 1})

    def test_video_logit_validation_is_stable_and_weights_partial_batches(self):
        class KnownLogits(nn.Module):
            def forward(self, values):
                assert not self.training
                assert not torch.is_grad_enabled()
                return values

        batches = [
            (torch.tensor([[1000., -1000.], [0., 0.]]), torch.tensor([1, 0])),
            (torch.tensor([[-1000., 1000.]]), torch.tensor([1])),
        ]
        metrics = _validate(KnownLogits(), batches, torch.device("cpu"), video_logits=True)
        expected = (2000.0 + torch.log(torch.tensor(2.)).item()) / 3
        self.assertAlmostEqual(metrics["loss"], expected, places=4)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["count"], 3)

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

    def test_early_stopping_uses_loss_even_when_accuracy_improves(self):
        validation = [
            {"loss": 1.0, "accuracy": 0.25, "count": 2},
            {"loss": 0.75, "accuracy": 0.25, "count": 2},
            {"loss": 0.75, "accuracy": 0.5, "count": 2},
            {"loss": 0.875, "accuracy": 0.75, "count": 2},
        ]
        self.config["training"].update(epochs=10, early_stopping_patience=2,
                                       early_stopping_min_delta=0.0)
        for name in ("per_frame", "late_fusion", "early_fusion", "cnn3d"):
            with self.subTest(model=name), contextlib.redirect_stdout(io.StringIO()) as output:
                self.config["model"]["name"] = name
                with patch("video_classification.train._validate", side_effect=validation) as validate:
                    run = train(self.config)
                self.assertEqual(validate.call_count, 4)
                self.assertIn("Early stopping at epoch 4", output.getvalue())
                summary = json.loads((run / "summary.json").read_text())
                self.assertEqual(summary["completed_epochs"], 4)
                self.assertEqual(summary["max_epochs"], 10)
                self.assertTrue(summary["stopped_early"])
                self.assertEqual(summary["stop_reason"], "early_stopping")
                self.assertEqual(summary["best_epoch"], 4)
                self.assertEqual(summary["best_loss_epoch"], 2)
                self.assertEqual(summary["minimum_val_video_loss"], 0.75)
                self.assertEqual(summary["early_stopping"]["bad_epochs"], 2)
                for filename, epoch in (("best.pt", 4), ("best_loss.pt", 2)):
                    checkpoint = torch.load(run / filename, map_location="cpu", weights_only=True)
                    self.assertEqual(checkpoint["epoch"], epoch)
                    model = build_model(checkpoint["config"])
                    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                self.assertEqual(self.config["training"]["epochs"], 10)

    def test_min_delta_resets_patience_only_on_significant_improvement(self):
        self.config["training"].update(epochs=10, early_stopping_patience=2,
                                       early_stopping_min_delta=0.125)
        # Binary-exact values: small drops accumulate, and equality to the
        # threshold does not reset patience. Still save the true lowest loss.
        validation = [{"loss": loss, "accuracy": 0.5, "count": 2}
                      for loss in (1.0, 0.9375, 0.8125, 0.75, 0.6875)]
        with contextlib.redirect_stdout(io.StringIO()):
            with patch("video_classification.train._validate", side_effect=validation):
                run = train(self.config)
        with (run / "history.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["early_stopping_bad_epochs"]) for row in rows], [0, 1, 0, 1, 2])
        summary = json.loads((run / "summary.json").read_text())
        self.assertEqual(summary["completed_epochs"], 5)
        self.assertEqual(summary["early_stopping"]["last_improvement_epoch"], 3)
        self.assertEqual(summary["early_stopping"]["reference_loss"], 0.8125)
        self.assertEqual(summary["best_loss_epoch"], 5)
        checkpoint = torch.load(run / "best_loss.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 5)
        self.assertEqual(checkpoint["validation"]["loss"], 0.6875)
        self.assertEqual(checkpoint["selection_metric"], "val_video_loss")

    def test_disabled_stopping_and_epoch_cap_finish_normally(self):
        validation = [{"loss": 1.0, "accuracy": 0.5, "count": 2}] * 3
        # With patience 2, the trigger coincides with the final allowed epoch.
        for patience in (0, 2):
            with self.subTest(patience=patience), contextlib.redirect_stdout(io.StringIO()) as output:
                self.config["training"]["early_stopping_patience"] = patience
                with patch("video_classification.train._validate", side_effect=validation):
                    run = train(self.config)
                summary = json.loads((run / "summary.json").read_text())
                self.assertEqual(summary["completed_epochs"], 3)
                self.assertFalse(summary["stopped_early"])
                self.assertEqual(summary["stop_reason"], "max_epochs")
                self.assertEqual(summary["early_stopping"]["enabled"], patience > 0)
                self.assertNotIn("Early stopping at epoch", output.getvalue())

    def test_invalid_early_stopping_settings_fail_before_loading_data(self):
        for key, values in (
            ("early_stopping_patience", (-1, 1.5, True, None)),
            ("early_stopping_min_delta", (-1, float("nan"), float("inf"), True, None)),
        ):
            for value in values:
                with self.subTest(setting=key, value=value):
                    config = copy.deepcopy(self.config)
                    config["training"][key] = value
                    with patch("video_classification.train.build_dataloaders") as loaders:
                        with self.assertRaisesRegex(ValueError, key):
                            train(config)
                        loaders.assert_not_called()
        template = (Path(__file__).resolve().parents[1] / "configs/01_per_frame.toml").read_text()
        config_path = self.root / "invalid.toml"
        for old, new in (("early_stopping_patience = 5", "early_stopping_patience = -1"),
                         ("early_stopping_min_delta = 0.0001", "early_stopping_min_delta = nan")):
            config_path.write_text(template.replace(old, new))
            with self.assertRaisesRegex(ValueError, "early_stopping"):
                load_config(config_path)

    def test_smoke_run_records_defaults_and_saves_loss_checkpoint(self):
        with contextlib.redirect_stdout(io.StringIO()):
            run = train(self.config, smoke_test=True)
        saved = json.loads((run / "config.json").read_text())
        self.assertEqual(saved["training"]["early_stopping_patience"], 5)
        self.assertEqual(saved["training"]["early_stopping_min_delta"], 0.0001)
        summary = json.loads((run / "summary.json").read_text())
        self.assertEqual(summary["completed_epochs"], 1)
        self.assertEqual(summary["max_epochs"], 1)
        self.assertFalse(summary["stopped_early"])
        self.assertTrue(summary["smoke_test"])
        checkpoint = torch.load(run / "best_loss.pt", map_location="cpu", weights_only=True)
        self.assertTrue(checkpoint["smoke_test"])
        self.assertEqual(checkpoint["validation"]["loss"], summary["minimum_val_video_loss"])

    def test_cli_overrides_early_stopping_settings(self):
        arguments = ["train", "--config", "unused.toml", "--epochs", "50",
                     "--patience", "0", "--min-delta", "0.002"]
        with patch("sys.argv", arguments):
            with patch("video_classification.train.load_config", return_value=self.config):
                with patch("video_classification.train.train") as run:
                    main()
        configured = run.call_args.args[0]["training"]
        self.assertEqual(configured["epochs"], 50)
        self.assertEqual(configured["early_stopping_patience"], 0)
        self.assertEqual(configured["early_stopping_min_delta"], 0.002)

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
