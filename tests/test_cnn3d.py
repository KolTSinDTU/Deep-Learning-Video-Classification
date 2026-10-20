"""Check temporal filtering, order sensitivity, and training of the 3D CNN."""

import unittest

import torch
from torch import nn

from video_classification.models import build_model
from video_classification.models.cnn3d import CNN3D


class CNN3DTests(unittest.TestCase):
    def test_factory_and_probabilities_support_different_clip_lengths(self):
        model = build_model({
            "model": {"name": "cnn3d"},
            "dataset": {"num_classes": 4},
        }).eval()
        self.assertIsInstance(model, CNN3D)
        for frames in (1, 2, 10):
            with self.subTest(frames=frames), torch.inference_mode():
                videos = torch.rand(2, 3, frames, 8, 16)
                logits = model(videos)
                probabilities = model.predict_video(videos)
                self.assertEqual(logits.shape, (2, 4))
                self.assertTrue(torch.isfinite(logits).all())
                torch.testing.assert_close(probabilities, logits.softmax(dim=1))
                torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))

    def test_temporal_filter_combines_neighbors_and_slides_over_time(self):
        model = CNN3D()
        first = model.features[0]
        with torch.no_grad():
            first.weight.zero_()
            first.bias.zero_()
            # Difference between the next and previous frame's red channel.
            first.weight[0, 0, 0, 1, 1] = -1
            first.weight[0, 0, 2, 1, 1] = 1
        for pulse, response in (
            (3, [0., 0., 1., 0., -1., 0., 0.]),
            (4, [0., 0., 0., 1., 0., -1., 0.]),
        ):
            with self.subTest(pulse=pulse), torch.inference_mode():
                videos = torch.zeros(1, 3, 7, 8, 8)
                videos[:, 0, pulse] = 1
                expected = torch.tensor(response)[:, None, None].expand(-1, 8, 8)
                torch.testing.assert_close(first(videos)[0, 0], expected)

    def test_full_model_can_distinguish_reversed_frame_order(self):
        model = CNN3D(num_classes=2).eval()
        convolutions = [layer for layer in model.features if isinstance(layer, nn.Conv3d)]
        with torch.no_grad():
            for layer in convolutions:
                layer.weight.zero_()
                layer.bias.zero_()
            convolutions[0].weight[0, 0, 0, 1, 1] = -1
            convolutions[0].weight[0, 0, 2, 1, 1] = 1
            for layer in convolutions[1:]:
                layer.weight[0, 0, 1, 1, 1] = 1
            model.classifier.weight.zero_()
            model.classifier.weight[:, 0] = torch.tensor([1., -1.])
            model.classifier.bias.copy_(torch.tensor([-1., 1.]))
        videos = torch.zeros(1, 3, 3, 8, 8)
        videos[:, 0] = torch.tensor([1., 2., 4.])[:, None, None]
        with torch.inference_mode():
            # ReLU keeps positive temporal differences before global pooling.
            forward = model(videos)
            backward = model(videos.flip(2))
        torch.testing.assert_close(forward, torch.tensor([[2 / 3, -2 / 3]]))
        torch.testing.assert_close(backward, torch.tensor([[-1 / 3, 1 / 3]]))
        self.assertEqual(forward.argmax(1).item(), 0)
        self.assertEqual(backward.argmax(1).item(), 1)

    def test_video_loss_updates_temporal_kernels_and_classifier(self):
        torch.manual_seed(42)
        model = CNN3D(num_classes=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        logits = model(torch.rand(2, 3, 3, 16, 24))
        loss = nn.functional.cross_entropy(logits, torch.tensor([0, 3]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        # Every temporal offset in the first kernel receives a learning signal.
        gradient = model.features[0].weight.grad
        self.assertTrue((gradient.abs().sum(dim=(0, 1, 3, 4)) > 0).all())
        optimizer.step()
        for name in ("features.0.weight", "classifier.weight"):
            self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]), name)

    def test_rejects_invalid_class_counts_and_clip_shapes(self):
        for count in (0, -1, 1.5, True):
            with self.subTest(classes=count), self.assertRaisesRegex(ValueError, "num_classes"):
                CNN3D(num_classes=count)
        model = CNN3D()
        for shape in ((2, 3, 16, 16), (2, 1, 3, 16, 16), (2, 3, 0, 16, 16),
                      (2, 3, 3, 7, 16), (2, 3, 3, 16, 7)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.rand(shape))


if __name__ == "__main__":
    unittest.main()
