import time
import torch

# 1. Importer votre module pour exécuter register_mjlab_task()
import src.tasks.crawling  # Ou: import src.tasks

from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv

task_name = "Mjlab-Crawling-Unitree-G1"

# 2. Charger la configuration
env_cfg = load_env_cfg(task_name)
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")

obs, _ = env.reset()
print("Robot initialisé avec succès ! Exécution de pas nuls...")

# 3. Tester avec des actions à zéro
for _ in range(200):
    actions = torch.zeros(env.action_space.shape, device=env.device)
    obs, reward, terminated, truncated, info = env.step(actions)
    time.sleep(0.02)

env.close()