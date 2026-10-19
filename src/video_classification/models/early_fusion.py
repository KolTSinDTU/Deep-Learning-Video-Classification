"""TODO: 2D CNN with frames fused at its input.

Input [B, 3, T, H, W]. Permute to [B, T, 3, H, W] and reshape to
[B, T*3, H, W]. Adapt the CNN's first convolution to 3*num_frames channels.
Document initialization if starting from pretrained RGB weights.
"""
