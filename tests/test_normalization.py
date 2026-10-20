"""Check optional normalization and compatibility with the original models."""

import unittest

import torch
from torch import nn

from video_classification.models import build_model


class NormalizationTests(unittest.TestCase):
    def make_config(self, name, normalization="group"):
        return {
            "model": {"name": name, "normalization": normalization},
            "dataset": {"num_classes": 4, "num_frames": 3},
        }

    def test_groupnorm_learns_with_one_sample_for_all_four_models(self):
        for name in ("per_frame", "late_fusion", "early_fusion", "cnn3d"):
            with self.subTest(model=name):
                torch.manual_seed(42)
                model = build_model(self.make_config(name))
                shape = (1, 3, 16, 16) if name == "per_frame" else (1, 3, 3, 16, 16)
                loss = nn.functional.cross_entropy(model(torch.rand(shape)), torch.tensor([2]))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                norms = [layer for layer in model.modules() if isinstance(layer, nn.GroupNorm)]
                self.assertEqual(len(norms), 3)
                for layer in norms:
                    self.assertTrue(torch.isfinite(layer.weight.grad).all())
                    self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_predictions_do_not_depend_on_other_samples_or_train_eval_mode(self):
        for name in ("per_frame", "late_fusion", "early_fusion", "cnn3d"):
            with self.subTest(model=name), torch.inference_mode():
                model = build_model(self.make_config(name))
                shape = (2, 3, 16, 16) if name == "per_frame" else (2, 3, 3, 16, 16)
                samples = torch.rand(shape)
                alone = model(samples[:1])
                together = model(samples)[:1]
                torch.testing.assert_close(alone, together, atol=1e-6, rtol=1e-5)
                model.eval()
                torch.testing.assert_close(model(samples[:1]), alone)

    def test_existing_parameters_and_seeded_initialization_are_preserved(self):
        for name in ("per_frame", "late_fusion", "early_fusion", "cnn3d"):
            with self.subTest(model=name):
                torch.manual_seed(42)
                baseline = build_model(self.make_config(name, "none"))
                torch.manual_seed(42)
                normalized = build_model(self.make_config(name))
                for key, value in baseline.state_dict().items():
                    torch.testing.assert_close(value, normalized.state_dict()[key])
                # Missing normalization means the original architecture, including keys.
                legacy_config = self.make_config(name)
                del legacy_config["model"]["normalization"]
                legacy = build_model(legacy_config)
                legacy.load_state_dict(baseline.state_dict(), strict=True)
                self.assertFalse(any(isinstance(layer, nn.GroupNorm) for layer in legacy.modules()))

    def test_shared_2d_encoder_still_matches_mean_frame_logits(self):
        config = self.make_config("per_frame")
        frame_model = build_model(config).eval()
        config["model"]["name"] = "late_fusion"
        late = build_model(config).eval()
        late.load_state_dict(frame_model.state_dict(), strict=True)
        videos = torch.rand(2, 3, 3, 16, 16)
        frames = videos.permute(0, 2, 1, 3, 4).reshape(6, 3, 16, 16)
        with torch.inference_mode():
            expected = frame_model(frames).reshape(2, 3, 4).mean(dim=1)
            torch.testing.assert_close(late(videos), expected)

    def test_unsupported_normalization_is_rejected(self):
        for name in ("per_frame", "late_fusion", "early_fusion", "cnn3d"):
            with self.subTest(model=name), self.assertRaisesRegex(ValueError, "normalization"):
                build_model(self.make_config(name, "unknown"))


if __name__ == "__main__":
    unittest.main()
