#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
from collections import OrderedDict

def flatten_state_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, OrderedDict)):
            items.extend(flatten_state_dict(v, new_key, sep=sep).items())
        elif isinstance(v, torch.Tensor):
            items.append((new_key, v))
    return dict(items)

def main():
    checkpoint_path = os.path.expanduser(
        "/home/noar/G1-Unitree-Ros2-Docker/src/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-09-03_16-34-24/model_49999.pt"
    )

    output_dir = os.path.expanduser("~/G1-Unitree-Ros2-Docker/src/g1_exported_policies")
    os.makedirs(output_dir, exist_ok=True)

    existing_files = os.listdir(output_dir)
    indices = []
    for f in existing_files:
        if f.startswith("mj_policy_") and f.endswith(".pt"):
            parts = f.split("_")
            if len(parts) == 2:
                idx_str = parts[1].split(".")[0]
                if idx_str.isdigit():
                    indices.append(int(idx_str))

    next_idx = max(indices) + 1 if indices else 0
    base_filename = f"mj_policy_{next_idx}"

    output_file_pt = os.path.join(output_dir, f"{base_filename}.pt")
    output_file_onnx = os.path.join(output_dir, f"{base_filename}.onnx")

    print(f"Chargement du checkpoint : {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "actor_state_dict" in checkpoint:
        raw_actor_dict = checkpoint["actor_state_dict"]
    else:
        raise RuntimeError("La clé 'actor_state_dict' est introuvable dans le checkpoint.")

    flat_actor_dict = flatten_state_dict(raw_actor_dict)

    # ── NOUVEAU : extraction de la normalisation d'observation ──
    norm_mean_key = next((k for k in flat_actor_dict if k.endswith("obs_normalizer._mean")), None)
    norm_std_key = next((k for k in flat_actor_dict if k.endswith("obs_normalizer._std")), None)

    if norm_mean_key is None or norm_std_key is None:
        raise RuntimeError(
            "obs_normalizer._mean / _std introuvables dans le checkpoint — "
            "vérifie que obs_normalization=true était bien actif à l'entraînement."
        )

    obs_mean = flat_actor_dict[norm_mean_key].clone().float()
    obs_std = flat_actor_dict[norm_std_key].clone().float()
    print(f"Normalisation trouvée : mean shape={obs_mean.shape}, std shape={obs_std.shape}")

    weight_shapes = []
    for k, v in flat_actor_dict.items():
        if v.ndim == 2 and "weight" in k.lower():
            weight_shapes.append((k, v.shape))

    print("\nMatrices de poids de l'acteur détectées :")
    for k, shape in weight_shapes:
        print(f"  - {k}: {shape}")

    if not weight_shapes:
        raise RuntimeError("Impossible de trouver des matrices de poids valides pour l'acteur.")

    weight_shapes = sorted(weight_shapes, key=lambda x: x[0])

    first_w_shape = weight_shapes[0][1]
    hidden_dim, input_dim = first_w_shape
    output_dim = weight_shapes[-1][1][0]

    print(f"\n✨ Dimensions extraites :")
    print(f"   - Entrée (input_dim) : {input_dim}")
    print(f"   - Couche cachée : {hidden_dim}")
    print(f"   - Sortie (output_dim) : {output_dim}")

    dims = [input_dim]
    for _, shape in weight_shapes:
        dims.append(shape[0])

    # ── NOUVEAU : wrapper qui normalise AVANT le MLP ──
    class NormalizedActorMLP(nn.Module):
        def __init__(self, layer_dims, obs_mean, obs_std):
            super().__init__()
            layers = []
            for i in range(len(layer_dims) - 2):
                layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
                layers.append(nn.ELU(alpha=1.0))
            layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
            self.mlp = nn.Sequential(*layers)
            self.register_buffer("obs_mean", obs_mean)
            self.register_buffer("obs_std", obs_std)

        def forward(self, x):
            x = (x - self.obs_mean) / self.obs_std
            return self.mlp(x)

    target_state_dict = {}
    layer_idx = 0
    for i in range(len(weight_shapes)):
        w_key, _ = weight_shapes[i]
        b_key = w_key.replace("weight", "bias")

        target_state_dict[f"mlp.{layer_idx}.weight"] = flat_actor_dict[w_key]
        if b_key in flat_actor_dict:
            target_state_dict[f"mlp.{layer_idx}.bias"] = flat_actor_dict[b_key]

        layer_idx += 2

    actor = NormalizedActorMLP(dims, obs_mean, obs_std)
    actor.load_state_dict(target_state_dict, strict=False)  # strict=False : obs_mean/std sont des buffers, pas dans target_state_dict
    actor.eval()

    dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)
    with torch.no_grad():
        out = actor(dummy_input)
    print(f"Test d'inférence réussi ! Shape de sortie : {out.shape}")

    traced_policy = torch.jit.trace(actor, dummy_input)
    traced_policy.save(output_file_pt)
    print(f"\n✅ SUCCÈS : Politique exportée en TorchScript vers : {output_file_pt}")

    torch.onnx.export(
        actor, dummy_input, output_file_onnx,
        export_params=True, opset_version=18, do_constant_folding=True,
        input_names=['input_obs'], output_names=['output_actions'],
        dynamic_axes={'input_obs': {0: 'batch_size'}, 'output_actions': {0: 'batch_size'}}
    )
    print(f"✅ SUCCÈS : Politique exportée en ONNX vers : {output_file_onnx}")
    print(f"👉 Kp/Kd/DEFAULT_POS/ACTION_SCALE inchangés dans g1_config.py. Pointe policy_path vers : {output_file_pt}")

if __name__ == "__main__":
    main()