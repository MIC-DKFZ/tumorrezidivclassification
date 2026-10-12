from pathlib import Path

import torch

from base_model import BaseModel


def _unwrap_state_dict(checkpoint):
    """Return a state_dict from common checkpoint container formats."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "network_weights", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _strip_common_prefixes(state_dict):
    cleaned = {}
    prefixes = ("module.", "model.", "net.")
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _adapt_first_conv_weight(weight, target_shape):
    if tuple(weight.shape) == tuple(target_shape):
        return weight

    if weight.ndim != 5:
        return None

    out_channels, in_channels, *kernel_size = weight.shape
    target_out, target_in, *target_kernel = target_shape
    if out_channels != target_out or kernel_size != target_kernel:
        return None

    if in_channels == 1 and target_in > 1:
        return weight.repeat(1, target_in, 1, 1, 1) / target_in

    if in_channels > target_in:
        return weight[:, :target_in]

    repeat_factor = target_in // in_channels
    remainder = target_in % in_channels
    parts = [weight.repeat(1, repeat_factor, 1, 1, 1)]
    if remainder:
        parts.append(weight[:, :remainder])
    return torch.cat(parts, dim=1) * (in_channels / target_in)


def _load_matching_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = _strip_common_prefixes(_unwrap_state_dict(checkpoint))
    target = model.state_dict()

    loaded = {}
    skipped = []
    adapted = []

    for key, value in source.items():
        if key not in target:
            skipped.append((key, "missing_in_target"))
            continue

        if tuple(value.shape) == tuple(target[key].shape):
            loaded[key] = value
            continue

        if key.endswith("conv1.weight"):
            adapted_weight = _adapt_first_conv_weight(value, target[key].shape)
            if adapted_weight is not None:
                loaded[key] = adapted_weight
                adapted.append(key)
                continue

        skipped.append((key, f"shape {tuple(value.shape)} -> {tuple(target[key].shape)}"))

    target.update(loaded)
    model.load_state_dict(target)

    print(
        "Loaded pretrained weights:",
        f"{len(loaded)} tensors from {checkpoint_path}",
        f"({len(adapted)} adapted, {len(skipped)} skipped)",
    )
    if adapted:
        print("Adapted tensors:", ", ".join(adapted))
    if skipped:
        print("First skipped pretrained tensors:")
        for key, reason in skipped[:20]:
            print(f"  {key}: {reason}")


class MonaiResNet18(BaseModel):
    def __init__(self, chpt_path=None, spatial_dims=3, **hypparams):
        super(MonaiResNet18, self).__init__(**hypparams)

        from monai.networks.nets import resnet18

        self.model = resnet18(
            pretrained=False,
            progress=True,
            spatial_dims=spatial_dims,
            n_input_channels=hypparams["input_channels"],
            num_classes=hypparams["num_classes"],
        )

        if hypparams["pretrained"]:
            if chpt_path is None:
                raise ValueError(
                    "MonaiResNet18 pretrained=True requires model.chpt_path. "
                    "Download MedicalNet/MONAI weights first and pass the local file path."
                )
            checkpoint_path = Path(chpt_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
            _load_matching_weights(self.model, checkpoint_path)

        # Dropout before the final classifier, matching the dropout rate reported
        # for the published model (model.resnet_dropout, e.g. 0.35).
        if self.resnet_dropout:
            in_features = self.model.fc.in_features
            self.model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=self.resnet_dropout),
                torch.nn.Linear(in_features, hypparams["num_classes"]),
            )

    def forward(self, x):
        return self.model(x)
