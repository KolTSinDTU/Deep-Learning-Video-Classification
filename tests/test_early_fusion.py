"""Check temporal channel stacking, order sensitivity, and video training."""

import unittest

import torch
from torch import nn

from video_classification.models import build_model
from video_classification.models.early_fusion import EarlyFusionCNN
from video_classification.models.per_frame import PerFrameCNN


class EarlyFusionCNNTests(unittest.TestCase):
    def test_first_convolution_receives_temporal_rgb_channels(self):
        model = build_model({
            "model": {"name": "early_fusion"},
            "dataset": {"num_classes": 4, "num_frames": 3},
        })
        self.assertIsInstance(model, EarlyFusionCNN)
        self.assertEqual(model.features[0].in_channels, 9)
        # Distinct colors, times, and videos catch incorrect channel packing.
        markers = torch.tensor([
            [[1., 11., 21.], [2., 12., 22.], [3., 13., 23.]],
            [[101., 111., 121.], [102., 112., 122.], [103., 113., 123.]],
        ])
        videos = markers[:, :, :, None, None].expand(-1, -1, -1, 8, 16)[..., ::2]
        observed = []
        handle = model.features[0].register_forward_pre_hook(
            lambda module, args: observed.append(args[0].detach().clone())
        )
        self.addCleanup(handle.remove)
        with torch.inference_mode():
            logits = model(videos)
        self.assertEqual(logits.shape, (2, 4))
        expected = torch.tensor([
            [1., 2., 3., 11., 12., 13., 21., 22., 23.],
            [101., 102., 103., 111., 112., 113., 121., 122., 123.],
        ])[:, :, None, None].expand(-1, -1, 8, 8)
        torch.testing.assert_close(observed[0], expected)

    def test_can_distinguish_reversed_frame_order(self):
        model = EarlyFusionCNN(num_classes=2, num_frames=2)
        model.features = nn.Sequential(
            nn.Conv2d(6, 1, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )
        model.classifier = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            model.features[0].weight.zero_()
            model.features[0].weight[0, 0, 0, 0] = 1
            model.features[0].weight[0, 3, 0, 0] = -1
            model.classifier.weight.copy_(torch.tensor([[1.], [-1.]]))
        videos = torch.zeros(1, 3, 2, 8, 8)
        videos[:, 0, 0] = 1
        videos[:, 0, 1] = 3
        torch.testing.assert_close(model(videos), torch.tensor([[-2., 2.]]))
        torch.testing.assert_close(model(videos.flip(2)), torch.tensor([[2., -2.]]))

    def test_video_loss_updates_all_frame_inputs_and_classifier(self):
        torch.manual_seed(42)
        model = EarlyFusionCNN(num_classes=4, num_frames=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        logits = model(torch.rand(2, 3, 3, 24, 16))
        loss = nn.functional.cross_entropy(logits, torch.tensor([0, 3]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        # Every temporal RGB block in the input convolution receives gradients.
        gradient = model.features[0].weight.grad.reshape(32, 3, 3, 3, 3)
        self.assertTrue((gradient.abs().sum(dim=(0, 2, 3, 4)) > 0).all())
        optimizer.step()
        for name in ("features.0.weight", "classifier.weight"):
            self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]), name)

    def test_single_frame_matches_per_frame_model_with_shared_weights(self):
        reference = PerFrameCNN(num_classes=4).eval()
        model = EarlyFusionCNN(num_classes=4, num_frames=1).eval()
        model.load_state_dict(reference.state_dict(), strict=True)
        frames = torch.rand(2, 3, 16, 24)
        with torch.inference_mode():
            expected = reference(frames)
            torch.testing.assert_close(model(frames.unsqueeze(2)), expected)
            probabilities = model.predict_video(frames.unsqueeze(2))
            torch.testing.assert_close(probabilities, expected.softmax(dim=1))
            torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))

    def test_rejects_invalid_frame_counts_and_input_shapes(self):
        for count in (0, -1, 1.5, True):
            with self.subTest(num_frames=count), self.assertRaisesRegex(ValueError, "num_frames"):
                EarlyFusionCNN(num_frames=count)
        model = EarlyFusionCNN(num_frames=3)
        for shape in ((2, 3, 16, 16), (2, 1, 3, 16, 16), (2, 3, 0, 16, 16),
                      (2, 3, 2, 16, 16), (2, 3, 3, 7, 16)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.rand(shape))


if __name__ == "__main__":
    unittest.main()
