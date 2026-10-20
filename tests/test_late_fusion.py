"""Check shared encoding, feature fusion, and video-level gradients."""

import unittest

import torch
from torch import nn

from video_classification.models import build_model
from video_classification.models.late_fusion import LateFusionCNN
from video_classification.models.per_frame import PerFrameCNN


class LateFusionCNNTests(unittest.TestCase):
    def test_uses_same_backbone_and_matches_mean_logits_with_shared_weights(self):
        torch.manual_seed(3)
        reference = PerFrameCNN(num_classes=4).eval()
        model = build_model({"model": {"name": "late_fusion"}, "dataset": {"num_classes": 4}})
        self.assertIsInstance(model, LateFusionCNN)
        model.load_state_dict(reference.state_dict(), strict=True)
        model.eval()
        # A non-contiguous clip also exercises the frame reshaping.
        videos = torch.rand(2, 3, 5, 16, 40)[..., ::2]
        with torch.inference_mode():
            expected = torch.stack(
                [reference(videos[:, :, t]) for t in range(5)], dim=1
            ).mean(dim=1)
            torch.testing.assert_close(model(videos), expected)
            torch.testing.assert_close(model(videos.flip(2)), expected)
            torch.testing.assert_close(model(videos[:, :, :1]), reference(videos[:, :, 0]))

    def test_fuses_features_before_softmax_without_mixing_videos_or_channels(self):
        model = LateFusionCNN(num_classes=2)
        # Treat RGB means as known features, with the first two as class logits.
        model.features = nn.AdaptiveAvgPool2d(1)
        model.classifier = nn.Linear(3, 2)
        with torch.no_grad():
            model.classifier.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
            model.classifier.bias.zero_()
        videos = torch.zeros(2, 3, 3, 8, 8)
        values = torch.tensor([4.0, -1.5, -1.5])[:, None, None]
        videos[0, 0] = values
        videos[1, 1] = values
        expected_logits = torch.tensor([[1 / 3, 0.], [0., 1 / 3]])
        torch.testing.assert_close(model(videos), expected_logits)
        probabilities = model.predict_video(videos)
        torch.testing.assert_close(probabilities, expected_logits.softmax(dim=1))
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))
        self.assertEqual(probabilities.argmax(dim=1).tolist(), [0, 1])
        # Probability averaging would choose the opposite class for video 0.
        mean_frame_probability = torch.sigmoid(torch.tensor([4., -1.5, -1.5])).mean()
        self.assertLess(mean_frame_probability.item(), 0.5)

    def test_video_loss_updates_encoder_and_classifier(self):
        torch.manual_seed(42)
        model = LateFusionCNN(num_classes=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        logits = model(torch.rand(2, 3, 3, 24, 16))
        self.assertEqual(logits.shape, (2, 4))
        loss = nn.functional.cross_entropy(logits, torch.tensor([0, 3]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        for name in ("features.0.weight", "classifier.weight"):
            self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]), name)

    def test_rejects_invalid_video_shapes(self):
        model = LateFusionCNN()
        for shape in ((2, 3, 16, 16), (2, 1, 5, 16, 16),
                      (2, 3, 0, 16, 16), (2, 3, 5, 7, 16)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.rand(shape))


if __name__ == "__main__":
    unittest.main()
