"""TODO: shared 2D encoder with feature fusion and a video classifier.

Input [B, 3, T, H, W]. Encode frames using a shared CNN, pool across time,
then classify the pooled representation. Train using video labels. This baseline
uses feature fusion to distinguish it from per-frame prediction aggregation.
"""
