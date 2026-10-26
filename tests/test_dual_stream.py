"""Dual-stream gradients, pairing, preprocessing, and end-to-end checkpoints."""

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from video_classification.config import load_config, validate_dual_stream
from video_classification.data import build_dataloaders
from video_classification.models import build_model
from video_classification.models.dual_stream import DualStreamCNN
from video_classification.train import train


class DualStreamModelTests(unittest.TestCase):
    def test_both_independent_encoders_and_fusion_learn(self):
        for channels in (2, 3):
            for normalization in ("none", "group"):
                with self.subTest(channels=channels, normalization=normalization):
                    torch.manual_seed(42)
                    model = DualStreamCNN(4, flow_channels=channels, normalization=normalization)
                    rgb = torch.rand(2, 3, 3, 16, 32)[..., ::2]
                    flow = torch.rand(2, channels, 2, 16, 32)[..., ::2]
                    parameters = dict(model.named_parameters())
                    names = ("rgb_features.0.weight", "flow_features.0.weight", "classifier.weight")
                    before = {name: parameters[name].detach().clone() for name in names}
                    self.assertNotEqual(parameters[names[0]].data_ptr(), parameters[names[1]].data_ptr())
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                    logits = model(rgb, flow)
                    self.assertEqual(logits.shape, (2, 4))
                    nn.functional.cross_entropy(logits, torch.tensor([0, 3])).backward()
                    for parameter in parameters.values():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    optimizer.step()
                    for name in names:
                        self.assertFalse(torch.equal(before[name], parameters[name]), name)
                    model.eval()
                    with torch.inference_mode():
                        probabilities = model.predict_video(rgb, flow)
                        torch.testing.assert_close(probabilities, model(rgb, flow).softmax(1))
                        torch.testing.assert_close(probabilities.sum(1), torch.ones(2))

    def test_fusion_preserves_streams_and_video_identity(self):
        model = DualStreamCNN(2, flow_channels=2)
        model.rgb_features = nn.AdaptiveAvgPool2d(1)
        model.flow_features = nn.AdaptiveAvgPool2d(1)
        model.classifier = nn.Linear(5, 2, bias=False)
        with torch.no_grad():
            model.classifier.weight.copy_(torch.tensor([[1., 0., 0., 2., 0.], [0., 1., 0., 0., 3.]]))
        rgb = torch.zeros(2, 3, 3, 8, 8)
        flow = torch.zeros(2, 2, 2, 8, 8)
        rgb[0, 0] = torch.tensor([0., 3., 6.])[:, None, None]
        rgb[1, 1] = 4
        flow[0, 0] = torch.tensor([1., 3.])[:, None, None]
        flow[1, 1] = 2
        torch.testing.assert_close(model(rgb, flow), torch.tensor([[7., 0.], [0., 10.]]))

    def test_rejects_bad_shapes(self):
        model = DualStreamCNN()
        rgb, flow = torch.rand(2, 3, 3, 8, 8), torch.rand(2, 3, 2, 8, 8)
        for bad_rgb, bad_flow in ((rgb[:, :, 0], flow), (rgb[:, :2], flow),
                                  (rgb, flow[:, :, :0]), (rgb, flow[:1]),
                                  (rgb, flow[..., :7]), (rgb, flow[:, :2])):
            with self.subTest(rgb=bad_rgb.shape, flow=bad_flow.shape), self.assertRaises(ValueError):
                model(bad_rgb, bad_flow)


class DualStreamPipelineTests(unittest.TestCase):
    def setUp(self):
        output = Path(__file__).resolve().parents[1] / "outputs"
        output.mkdir(exist_ok=True)
        temporary = TemporaryDirectory(prefix="_test_dual_", dir=output)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.manifests = self.data / "metadata" / "flow"
        self.manifests.mkdir(parents=True)
        self.config = {
            "experiment": {"name": "dual_test", "seed": 42, "output_dir": str(self.root / "runs")},
            "dataset": {"root": str(self.data), "num_classes": 2, "num_frames": 3,
                        "image_size": "native", "regime": "no_leakage", "flow_format": "rgb_png",
                        "flow_manifest_root": str(self.manifests)},
            "model": {"name": "dual_stream"},
            "training": {"epochs": 2, "batch_size": 2, "num_workers": 0,
                         "learning_rate": 0.001, "device": "cpu"},
        }
        self.rows = {}
        for split in ("train", "val"):
            videos, pairs = [], []
            for label, action in enumerate(("squats", "jumps")):
                video = f"{split}_{action}"
                videos.append(dict(video_name=video, video_path=f"{action}/{video}.avi", action=action, label=label))
                frames = self.data / "frames" / split / action / video
                frames.mkdir(parents=True)
                for i in range(1, 4):
                    Image.new("RGB", (20, 16), (label * 70, i * 30, 40)).save(frames / f"frame_{i}.jpg")
                for i in (1, 2):
                    # Names deliberately do not encode temporal order.
                    relative = f"flows-png/{split}/{video}/{'z' if i == 1 else 'a'}.png"
                    path = self.data / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (20, 16), (i * 50, label * 70, 0)).save(path)
                    pairs.append(dict(video_name=video, from_frame=i, to_frame=i+1, flow_path=relative))
            self.write_csv(self.data / "metadata" / f"{split}.csv", videos)
            self.rows[split] = list(reversed(pairs))
            self.write_manifest(split)
        # No test metadata exists: the training pipeline must not access it.

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def write_manifest(self, split="train"):
        self.write_csv(self.manifests / f"{split}.csv", self.rows[split])

    def make_vectors(self):
        self.config["dataset"].update(flow_format="uv_npy", flow_scale=2.0)
        for split, rows in self.rows.items():
            for row in rows:
                row["flow_path"] = str(Path(row["flow_path"]).with_suffix(".npy"))
                values = np.zeros((16, 20, 2), dtype=np.float32)
                values[..., 0], values[..., 1] = 4., -2.
                np.save(self.data / row["flow_path"], values)
            self.write_manifest(split)

    def use_course_layout(self):
        self.config["dataset"].pop("flow_manifest_root")
        self.config["dataset"]["num_frames"] = 10
        for split in ("train", "val"):
            (self.manifests / f"{split}.csv").unlink()
            for label, action in enumerate(("squats", "jumps")):
                video = f"{split}_{action}"
                rgb_folder = self.data / "frames" / split / action / video
                flow_folder = self.data / "flows_png" / split / action / video
                flow_folder.mkdir(parents=True)
                for i in range(1, 11):
                    Image.new("RGB", (20, 16), (label * 70, i * 20, 40)).save(rgb_folder / f"frame_{i}.jpg")
                for i in range(1, 10):
                    Image.new("RGB", (20, 16), (i * 20, label * 70, 0)).save(flow_folder / f"flow_{i}_{i+1}.png")

    def test_course_layout_loads_ten_rgb_and_nine_flow_frames_without_manifests(self):
        self.use_course_layout()
        loaders = build_dataloaders(self.config)
        for split in ("train", "val"):
            dataset = loaders[split].dataset
            for index in range(2):
                clips, label = dataset[index]
                self.assertEqual(label, index)
                self.assertEqual(clips["rgb"].shape, (3, 10, 16, 20))
                self.assertEqual(clips["flow"].shape, (3, 9, 16, 20))
                torch.testing.assert_close(clips["flow"][0, :, 0, 0], torch.arange(1, 10) * (20 / 255))
                torch.testing.assert_close(clips["flow"][1, :, 0, 0], torch.full((9,), index * 70 / 255))
                self.assertEqual(dataset.flow_records[index][-1].name, "flow_9_10.png")

    def test_course_layout_reports_missing_flow_pair(self):
        self.use_course_layout()
        (self.data / "flows_png/train/squats/train_squats/flow_4_5.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "flow_4_5.png"):
            build_dataloaders(self.config)

    def test_course_layout_smoke_training_and_checkpoint_reload(self):
        self.use_course_layout()
        with contextlib.redirect_stdout(io.StringIO()):
            run = train(self.config, smoke_test=True)
        saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        rebuilt = build_model(saved["config"])
        rebuilt.load_state_dict(saved["model_state_dict"], strict=True)
        clips, _ = next(iter(build_dataloaders(saved["config"])["val"]))
        with torch.inference_mode():
            self.assertEqual(rebuilt.predict_video(**clips).shape, (1, 2))
        self.assertEqual(saved["validation"]["count"], 2)
        self.assertNotIn("flow_manifest_root", saved["config"]["dataset"])

    def test_pairing_uses_video_and_numeric_frame_pair_not_row_or_filename_order(self):
        loaders = build_dataloaders(self.config)
        clips, labels = next(iter(loaders["val"]))
        self.assertEqual(clips["rgb"].shape, (1, 3, 3, 16, 20))
        self.assertEqual(clips["flow"].shape, (1, 3, 2, 16, 20))
        self.assertEqual(labels.tolist(), [0])
        torch.testing.assert_close(clips["flow"][0, 0, :, 0, 0], torch.tensor([50., 100.]) / 255)

    def test_vectors_keep_sign_and_rescale_displacements_when_resizing(self):
        self.make_vectors()
        for size, expected in (("native", [2., -1.]), (8, [0.8, -0.5])):
            with self.subTest(size=size):
                self.config["dataset"]["image_size"] = size
                clips, _ = build_dataloaders(self.config)["train"].dataset[0]
                torch.testing.assert_close(clips["flow"][:, 0, 0, 0], torch.tensor(expected))

    def test_invalid_pairing_fails_before_training(self):
        original = copy.deepcopy(self.rows["train"])
        for problem in ("missing", "duplicate", "unknown", "nonconsecutive", "path", "file", "reused"):
            with self.subTest(problem=problem):
                self.rows["train"] = copy.deepcopy(original)
                row = self.rows["train"][0]
                if problem == "missing":
                    self.rows["train"].pop()
                elif problem == "duplicate":
                    self.rows["train"].append(dict(row))
                elif problem == "unknown":
                    row["video_name"] = "wrong_video"
                elif problem == "nonconsecutive":
                    row["to_frame"] = row["from_frame"]
                elif problem == "path":
                    row["flow_path"] = "../outside.png"
                elif problem == "file":
                    row["flow_path"] = "missing.png"
                else:
                    row["flow_path"] = self.rows["train"][1]["flow_path"]
                self.write_manifest()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    build_dataloaders(self.config)

    def test_rejects_shared_flow_file_across_splits(self):
        self.rows["val"][0]["flow_path"] = self.rows["train"][0]["flow_path"]
        self.write_manifest("val")
        with self.assertRaisesRegex(ValueError, "both train and validation"):
            build_dataloaders(self.config)

    def test_rejects_wrong_representation_and_dimensions(self):
        path = self.data / self.rows["train"][0]["flow_path"]
        for mode, size in (("L", (20, 16)), ("RGB", (12, 12))):
            with self.subTest(mode=mode, size=size):
                Image.new(mode, size).save(path)
                dataset = build_dataloaders(self.config)["train"].dataset
                with self.assertRaises(ValueError):
                    dataset[1]
        self.make_vectors()
        path = self.data / self.rows["train"][0]["flow_path"]
        for values in (np.zeros((16, 20, 2), dtype=np.uint8),
                       np.full((16, 20, 2), np.nan), np.zeros((16, 20, 3))):
            np.save(path, values)
            with self.assertRaisesRegex(ValueError, "finite floating-point"):
                build_dataloaders(self.config)["train"].dataset[1]

    def test_native_cache_requires_matching_source_pair_indices(self):
        self.config["dataset"]["frame_source"] = "native_video"
        for split in ("train", "val"):
            for action in ("squats", "jumps"):
                folder = self.data / "frames_native" / split / action / f"{split}_{action}"
                folder.mkdir(parents=True)
                for i in range(1, 4):
                    Image.new("RGB", (20, 16)).save(folder / f"frame_{i}.png")
                (folder / "sampling.json").write_text(json.dumps(
                    {"num_frames": 3, "frame_indices_zero_based": [0, 5, 10]}))
        custom_manifest = self.config["dataset"].pop("flow_manifest_root")
        with self.assertRaisesRegex(ValueError, "custom/native flow needs"):
            build_dataloaders(self.config)
        self.config["dataset"]["flow_manifest_root"] = custom_manifest
        with self.assertRaisesRegex(ValueError, "source_from_index"):
            build_dataloaders(self.config)
        for split, rows in self.rows.items():
            for row in rows:
                row.update(source_from_index=(row["from_frame"]-1)*5,
                           source_to_index=(row["to_frame"]-1)*5)
            self.write_manifest(split)
        build_dataloaders(self.config)
        self.rows["train"][0]["source_to_index"] = 999
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "source indices disagree"):
            build_dataloaders(self.config)

    def test_smoke_train_checkpoint_reload_and_video_metrics_for_both_formats(self):
        for format_name in ("rgb_png", "uv_npy"):
            with self.subTest(format=format_name):
                if format_name == "uv_npy":
                    self.make_vectors()
                    self.config["model"]["normalization"] = "group"
                with contextlib.redirect_stdout(io.StringIO()):
                    run = train(self.config, smoke_test=True)
                for filename in ("best.pt", "best_loss.pt"):
                    saved = torch.load(run / filename, map_location="cpu", weights_only=True)
                    rebuilt = build_model(saved["config"])
                    rebuilt.load_state_dict(saved["model_state_dict"], strict=True)
                    self.assertTrue(saved["smoke_test"])
                    self.assertEqual(saved["validation"]["count"], 2)
                    clips, _ = next(iter(build_dataloaders(saved["config"])["val"]))
                    with torch.inference_mode():
                        self.assertEqual(rebuilt.predict_video(**clips).shape, (1, 2))
                with (run / "history.csv").open() as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(int(row["train_videos"]), 2)
                self.assertNotIn("train_frames", row)
                environment = json.loads((run / "environment.json").read_text())
                self.assertEqual(environment["flow_format"], format_name)
        self.assertEqual(self.config["training"]["epochs"], 2)

    def test_config_requires_corrected_regime_and_explicit_flow_format_and_scale(self):
        valid = self.config["dataset"]
        for changes in ({"regime": "original"}, {"flow_format": None}, {"num_frames": 1},
                        {"flow_format": "uv_npy"}, {"flow_format": "uv_npy", "flow_scale": 0},
                        {"flow_format": "uv_npy", "flow_scale": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_dual_stream({**valid, **changes})
        config = load_config(Path(__file__).resolve().parents[1] / "configs/06_dual_stream.toml")
        self.assertEqual(config["dataset"]["frame_source"], "provided")
        self.assertNotIn("flow_manifest_root", config["dataset"])


if __name__ == "__main__":
    unittest.main()
