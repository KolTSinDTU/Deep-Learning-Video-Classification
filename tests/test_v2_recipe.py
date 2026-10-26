"""v2 recipe: shared backbones, weight transfer, clip augmentation, flow stream, fusion."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import torch
from torch import nn

from video_classification.data import build_dataloaders
from video_classification.evaluate import load_checkpoint, predict
from video_classification.fuse import fuse
from video_classification.initialization import _transfer, initialize_model
from video_classification.models import build_model
from video_classification.train import build_scheduler, train
from video_classification.transforms import ClipTransform

from test_dual_stream import DualStreamPipelineTests

MAPPING = {"a": 0, "b": 1, "c": 2}


def _config(name, **model):
    return {"model": {"name": name, "backbone": "vgg_bn", **model},
            "dataset": {"num_classes": 3, "num_frames": 4, "regime": "original",
                        "flow_format": "rgb_png"},
            "training": {}}


class BackboneTransferTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        torch.manual_seed(0)
        self.source = build_model(_config("per_frame")).eval()
        # Non-trivial BN statistics so that transfer of buffers is checked too.
        for module in self.source.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.uniform_(-0.1, 0.1)
                module.running_var.uniform_(0.5, 1.5)
        self.path = Path(temporary.name) / "per_frame.pt"
        torch.save({"config": _config("per_frame"), "class_to_label": MAPPING,
                    "model_state_dict": self.source.state_dict(), "epoch": 3,
                    "smoke_test": False}, self.path)

    def transferred(self, name, **model):
        config = _config(name, **model)
        config["training"]["init_checkpoint"] = str(self.path)
        target = build_model(config)
        provenance = initialize_model(target, config, MAPPING)
        self.assertEqual(provenance["method"], "per_frame_checkpoint")
        return target.eval()

    def test_early_fusion_inflation_reproduces_per_frame_logits_on_static_clip(self):
        target = self.transferred("early_fusion")
        frame = torch.rand(2, 3, 32, 32)
        clip = frame[:, :, None].expand(-1, -1, 4, -1, -1)
        with torch.inference_mode():
            torch.testing.assert_close(target(clip), self.source(frame), rtol=1e-4, atol=1e-5)

    def test_cnn3d_inflation_matches_2d_kernels_summed_over_time(self):
        target = self.transferred("cnn3d")
        convolutions = [m for m in target.features if isinstance(m, nn.Conv3d)]
        sources = [m for m in self.source.features if isinstance(m, nn.Conv2d)]
        self.assertEqual(len(convolutions), len(sources))
        for conv3d, conv2d in zip(convolutions, sources):
            torch.testing.assert_close(conv3d.weight.sum(dim=2), conv2d.weight)
        with torch.inference_mode():
            self.assertEqual(target(torch.rand(1, 3, 10, 32, 32)).shape, (1, 3))

    def test_late_fusion_and_dual_stream_rgb_encoder_copy_weights(self):
        late = self.transferred("late_fusion", temporal_pool="max")
        for key, value in self.source.state_dict().items():
            torch.testing.assert_close(late.state_dict()[key], value)
        config = _config("dual_stream", flow_input="stack")
        config["dataset"]["regime"] = "no_leakage"
        dual = build_model(config)
        flow_before = {k: v.clone() for k, v in dual.flow_features.state_dict().items()}
        _transfer(dual, "dual_stream", self.source.state_dict())
        for key, value in self.source.features.state_dict().items():
            torch.testing.assert_close(dual.rgb_features.state_dict()[key], value)
        for key, value in flow_before.items():
            torch.testing.assert_close(dual.flow_features.state_dict()[key], value)

    def test_rejects_backbone_mismatch(self):
        config = _config("late_fusion", backbone="small")
        config["training"]["init_checkpoint"] = str(self.path)
        with self.assertRaisesRegex(ValueError, "same backbone"):
            initialize_model(build_model(config), config, MAPPING)

    def test_stacked_flow_stream_sees_all_fields_in_first_layer(self):
        config = _config("flow_stream")
        config["dataset"]["regime"] = "no_leakage"
        model = build_model(config)
        self.assertEqual(model.features[0].in_channels, 9)  # 3 channels x (4 - 1) fields
        self.assertEqual(model(torch.rand(2, 3, 3, 32, 32)).shape, (2, 3))
        with self.assertRaisesRegex(ValueError, "Expected 3 frames"):
            model(torch.rand(2, 3, 2, 32, 32))


class RecipeTests(unittest.TestCase):
    def test_cosine_schedule_warms_up_then_decays_to_zero(self):
        optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = build_scheduler(optimizer, {"scheduler": "cosine", "epochs": 10,
                                                "warmup_epochs": 2})
        rates = []
        for _ in range(10):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(rates[0], 1 / 3)
        self.assertAlmostEqual(rates[2], 1.0)
        self.assertTrue(all(a >= b for a, b in zip(rates[2:], rates[3:])))
        self.assertLess(rates[-1], 0.05)

    def test_augmentation_is_identical_across_frames_of_a_clip(self):
        transform = ClipTransform(16, augment=True, normalization="standardize")
        image = Image.frombytes("RGB", (40, 30), bytes(i * 7 % 256 for i in range(3600)))
        for seed in range(5):
            torch.manual_seed(seed)
            params = transform.sample(image)
            first, second = transform.rgb(image, params), transform.rgb(image, params)
            self.assertEqual(first.shape, (3, 16, 16))
            torch.testing.assert_close(first, second)

    def test_vector_flow_flip_negates_horizontal_motion_and_rescales(self):
        transform = ClipTransform(10, augment=False)
        flow = torch.stack([torch.full((20, 40), 4.0), torch.full((20, 40), -2.0)])
        params = {"crop": None, "flip": True, "jitter": None}
        torch.testing.assert_close(transform.flow_uv(flow, params)[:, 0, 0], torch.tensor([-1.0, -1.0]))
        with self.assertRaisesRegex(ValueError, "colour-coded"):
            transform.flow_png(Image.new("RGB", (40, 20)), params)

    def test_standardized_colour_flow_maps_zero_motion_to_zero(self):
        transform = ClipTransform(8, normalization="standardize")
        white = Image.new("RGB", (16, 16), (255, 255, 255))
        params = transform.sample(white)
        torch.testing.assert_close(transform.flow_png(white, params), torch.zeros(3, 8, 8))


class FlowStreamPipelineTests(unittest.TestCase):
    setUp = DualStreamPipelineTests.setUp
    write_csv = staticmethod(DualStreamPipelineTests.write_csv)
    write_manifest = DualStreamPipelineTests.write_manifest
    use_course_layout = DualStreamPipelineTests.use_course_layout

    def v2(self, name, **model):
        self.config["dataset"].update(image_size=16, augment=True, input_normalization="standardize")
        self.config["model"] = {"name": name, "backbone": "vgg_bn", "dropout": 0.5, **model}
        self.config["training"].update(epochs=1, optimizer="adamw", weight_decay=0.05,
                                       scheduler="cosine", validation_batch_size=2)
        self.config["experiment"]["output_dir"] = str(self.root / "runs" / name)

    def test_flow_stream_loads_flow_only_without_flips(self):
        self.use_course_layout()
        self.v2("flow_stream")
        dataset = build_dataloaders(self.config)["train"].dataset
        self.assertFalse(dataset.clip_transform.allow_flip)
        sample, _ = dataset[0]
        self.assertEqual(set(sample), {"flow"})
        self.assertEqual(sample["flow"].shape, (3, 9, 16, 16))

    def test_stream_training_evaluation_and_score_fusion(self):
        self.use_course_layout()
        runs = {}
        for name, model in (("flow_stream", {}), ("late_fusion", {}),
                            ("dual_stream", {"flow_input": "stack"})):
            self.v2(name, **model)
            with contextlib.redirect_stdout(io.StringIO()):
                runs[name] = train(self.config) / "best.pt"
            environment = json.loads((runs[name].parent / "environment.json").read_text())
            self.assertIn("random resized crop", environment["augmentation"])
        model, config, _ = load_checkpoint(runs["dual_stream"])
        result = predict(model, config, "val", torch.device("cpu"))
        self.assertEqual(result["probabilities"].shape, (2, 2))
        with contextlib.redirect_stdout(io.StringIO()):
            report = fuse(runs["late_fusion"], runs["flow_stream"], self.root / "fusion",
                          device="cpu")
        self.assertIn(report["selected_flow_weight"], [row["flow_weight"]
                                                       for row in report["val_weight_sweep"]])
        self.assertEqual(set(report["results"]), {"val"})
        self.assertTrue((self.root / "fusion" / "fusion_val.json").is_file())


if __name__ == "__main__":
    unittest.main()
