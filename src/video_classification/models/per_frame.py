"""TODO: 2D per-frame classifier and video prediction aggregation.

Train on individual RGB frames [B, 3, H, W]. For video evaluation, predict all
sampled frames, then aggregate their probabilities per video (one possible
baseline). Document the aggregation rule. Return video scores [B, num_classes]
and compute video-level metrics, not frame-level test accuracy.
"""
