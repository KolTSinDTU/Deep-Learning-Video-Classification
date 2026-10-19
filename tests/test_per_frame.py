"""Behavior checks for frame training and video probability aggregation."""

import unittest

import torch
from torch import nn

from video_classification.models import build_model
from video_classification.models.per_frame import PerFrameCNN


class PerFrameCNNTests(unittest.TestCase):
    def test_frame_training_updates_encoder_and_classifier(self):
        torch.manual_seed(42)
        model = build_model({"model": {"name": "per_frame"}, "dataset": {"num_classes": 4}})
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        frames = torch.rand(3, 3, 32, 40)
        labels = torch.tensor([0, 1, 3])
        before = {name: param.detach().clone() for name, param in model.named_parameters()}

        logits = model(frames)
        self.assertEqual(logits.shape, (3, 4))
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        optimizer.step()
        for name in ("features.0.weight", "classifier.weight"):
            self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]), name)

    def test_video_aggregation_preserves_frames_and_averages_probabilities(self):
        class PixelClassifier(PerFrameCNN):
            """Treat the first two RGB values as known logits for each frame."""

            def forward(self, frames):
                return frames[:, :2, 0, 0]

        model = PixelClassifier(num_classes=2).eval()
        # Distinct clips/channels/times expose incorrect flattening or averaging.
        videos = torch.zeros(2, 3, 3, 8, 8)
        videos[0, 0, :, :, :] = torch.tensor([4.0, -1.5, -1.5])[:, None, None]
        videos[1, 1, :, :, :] = torch.tensor([0.0, 2.0, 4.0])[:, None, None]
        expected_first_class = torch.stack([
            torch.sigmoid(torch.tensor([4.0, -1.5, -1.5])).mean(),
            torch.sigmoid(torch.tensor([0.0, -2.0, -4.0])).mean(),
        ])
        expected = torch.stack((expected_first_class, 1 - expected_first_class), dim=1)

        actual = model.predict_video(videos)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(dim=1), torch.ones(2))
        # Averaging logits first would predict class 0 for the first clip.
        self.assertEqual(actual[0].argmax().item(), 1)

    def test_real_cnn_video_predictions_match_individual_frame_predictions(self):
        torch.manual_seed(7)
        model = PerFrameCNN(num_classes=3).eval()
        videos = torch.rand(2, 3, 5, 16, 24)
        with torch.inference_mode():
            expected = torch.stack([
                torch.stack([model(clip[:, t].unsqueeze(0)).softmax(-1)[0]
                             for t in range(clip.shape[1])]).mean(0)
                for clip in videos
            ])
            torch.testing.assert_close(model.predict_video(videos), expected)
            torch.testing.assert_close(model.predict_video(videos.flip(2)), expected)
            torch.testing.assert_close(
                model.predict_video(videos[:, :, :1]), model(videos[:, :, 0]).softmax(-1)
            )

    def test_rejects_invalid_input_shapes(self):
        model = PerFrameCNN()
        for shape in ((2, 1, 16, 16), (2, 3, 4, 16, 16), (2, 3, 7, 16)):
            with self.subTest(frame_shape=shape), self.assertRaises(ValueError):
                model(torch.rand(shape))
        for shape in ((2, 3, 16, 16), (2, 1, 4, 16, 16), (2, 3, 0, 16, 16)):
            with self.subTest(video_shape=shape), self.assertRaises(ValueError):
                model.predict_video(torch.rand(shape))


if __name__ == "__main__":
    unittest.main()
