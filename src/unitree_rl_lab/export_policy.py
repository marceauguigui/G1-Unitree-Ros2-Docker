#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
from collections import OrderedDict

# ── 1. Define the exact Actor MLP architecture matching your training ──
class G1ActorMLP(nn.Module):
    def __init__(self, in_features=1415, out_features=29):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ELU(alpha=1.0),
            nn.Linear(512, 256),
            nn.ELU(alpha=1.0),
            nn.Linear(256, 128),
            nn.ELU(alpha=1.0),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        return self.mlp(x)


def flatten_dict(d, parent_key='', sep='.'):
    """Recursively flattens nested dictionaries into single key-tensor pairs."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, OrderedDict)) or hasattr(v, 'items'):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, torch.Tensor):
            items.append((new_key, v))
    return dict(items)


def main():
    # ── 2. Configure file paths ──
    checkpoint_path = os.path.expanduser(
        "~/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-07-28_18-16-37/model_11300.pt"
    )
    output_dir = os.path.expanduser("~/g1_policy")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_pt = os.path.join(output_dir, "policy2.pt")
    output_file_onnx = os.path.join(output_dir, "policy2.onnx")

    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Flatten nested dicts to get pure Tensors
    flat_tensors = flatten_dict(checkpoint)

    # Expected tensor dimensions in sequence: (weight_shape, bias_shape)
    expected_shapes = [
        (torch.Size([512, 1415]), torch.Size([512])),
        (torch.Size([256, 512]), torch.Size([256])),
        (torch.Size([128, 256]), torch.Size([128])),
        (torch.Size([29, 128]), torch.Size([29]))
    ]

    # Filter out Critic weights
    actor_tensors = {k: v for k, v in flat_tensors.items() if "critic" not in k.lower()}

    target_keys = [
        "mlp.0.weight", "mlp.0.bias",
        "mlp.2.weight", "mlp.2.bias",
        "mlp.4.weight", "mlp.4.bias",
        "mlp.6.weight", "mlp.6.bias"
    ]
    
    actor_state_dict = {}
    used_keys = set()

    # Smart mapping by matching tensor dimensions
    for layer_idx, (w_shape, b_shape) in enumerate(expected_shapes):
        w_key, b_key = None, None
        
        for k, tensor in actor_tensors.items():
            if k not in used_keys and tensor.shape == w_shape:
                w_key = k
                break
        for k, tensor in actor_tensors.items():
            if k not in used_keys and k != w_key and tensor.shape == b_shape:
                b_key = k
                break

        if w_key and b_key:
            used_keys.add(w_key)
            used_keys.add(b_key)
            actor_state_dict[target_keys[layer_idx * 2]] = actor_tensors[w_key]
            actor_state_dict[target_keys[layer_idx * 2 + 1]] = actor_tensors[b_key]
            print(f"Mapped Layer {layer_idx}: {w_key}")
        else:
            print(f"❌ Could not match shapes for Layer {layer_idx} ({w_shape}, {b_shape})")

    # Load into clean actor model
    actor = G1ActorMLP(in_features=1415, out_features=29)
    actor.load_state_dict(actor_state_dict)
    actor.eval()

    dummy_input = torch.zeros(1, 1415, dtype=torch.float32)

    # 1. Trace and export to TorchScript (.pt)
    traced_policy = torch.jit.trace(actor, dummy_input)
    traced_policy.save(output_file_pt)
    print(f"\n✅ SUCCESS: TorchScript saved at: {output_file_pt}")

    # 2. Export to ONNX (.onnx)
    torch.onnx.export(
        actor,
        dummy_input,
        output_file_onnx,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_obs'],
        output_names=['output_actions'],
        dynamic_axes={
            'input_obs': {0: 'batch_size'},
            'output_actions': {0: 'batch_size'}
        }
    )
    print(f"✅ SUCCESS: ONNX model saved at: {output_file_onnx}")


if __name__ == "__main__":
    main()