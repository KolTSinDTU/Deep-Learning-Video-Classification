"""Model factory for RGB, flow, and dual-stream video classifiers."""

from ..config import MODEL_NAMES


def build_model(config: dict):
    """Construct the selected model with dataset.num_classes output logits."""
    model, data = config["model"], config["dataset"]
    name = model["name"]
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    model_options = {
        "num_classes": data["num_classes"],
        "normalization": model.get("normalization", "none"),
        "backbone": model.get("backbone", "small"),
        "dropout": model.get("dropout", 0.0),
    }
    pretrained = model.get("pretrained", False)
    if name == "per_frame":
        from .per_frame import PerFrameCNN

        return PerFrameCNN(**model_options, pretrained=pretrained)
    if name == "late_fusion":
        from .late_fusion import LateFusionCNN

        return LateFusionCNN(**model_options, pretrained=pretrained,
                             temporal_pool=model.get("temporal_pool", "mean"))
    if name == "early_fusion":
        from .early_fusion import EarlyFusionCNN

        return EarlyFusionCNN(**model_options, pretrained=pretrained,
                              num_frames=data["num_frames"])
    if name == "cnn3d":
        from .cnn3d import CNN3D

        if pretrained:
            raise ValueError("cnn3d has no pretrained weights; use training.init_checkpoint "
                             "to inflate a trained vgg_bn per-frame model")
        return CNN3D(**model_options)
    if name in {"dual_stream", "flow_stream"}:
        from ..config import validate_dual_stream
        from .dual_stream import DualStreamCNN, FlowStreamCNN

        validate_dual_stream(data)
        flow_options = {
            "flow_channels": 3 if data["flow_format"] == "rgb_png" else 2,
            "num_flow_frames": data["num_frames"] - 1,
            "pretrained": pretrained,
        }
        if name == "flow_stream":
            return FlowStreamCNN(**model_options, **flow_options)
        return DualStreamCNN(**model_options, **flow_options,
                             flow_input=model.get("flow_input", "mean"))
    raise NotImplementedError(f"Implement {name} in models/{name}.py and register it here")
