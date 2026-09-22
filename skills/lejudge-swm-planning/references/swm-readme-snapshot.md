# stable-worldmodel README — snapshot 2026-09-22 (verify against installed version)

Install: `pip install 'stable-worldmodel[all]'` (base + training, environments, data formats). Python 3.10 venv via uv recommended.

Quick start (verbatim shape):
```python
import stable_worldmodel as swm
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.solver import CEMSolver
world = swm.World("swm/PushT-v1", num_envs=8)
solver = CEMSolver(model=world_model, num_samples=300)
policy = WorldModelPolicy(solver=solver, config=PlanConfig(horizon=10, receding_horizon=5))
world.set_policy(policy); results = world.evaluate(episodes=50)
```

Solvers: CEM, iCEM, MPPI, Predictive Sampling, SGD/Adam, PGD, Augmented Lagrangian.
Baselines: DINO-WM, PLDM, LeWM, GCBC, GCIVL, GCIQL (`scripts/train/`).
Envs relevant here: `swm/PushT-v1` (16 FoVs), `swm/TwoRoom-v1`, `swm/OGBCube-v0`, `swm/ReacherDMControl-v0`.
CLI: `swm datasets`, `swm inspect <dataset>`, `swm envs`, `swm fovs PushT-v1`, `swm checkpoints`.
Data formats: lance (default), hdf5, folder, video, lerobot.

LeWM checkpoints (HF): quentinll/lewm-pusht, -cube, -tworooms, -reacher. Loading: `swm.policy.AutoCostModel('pusht/lewm')` after converting `weights.pt`+`config.json` to `lewm_object.ckpt` (script in le-wm README).

LeWM planning: CEM minimising `‖ẑ_H − z_g‖²`, MPC executes first K actions then replans; planning under 1 s on the paper's setup.

Sources: https://github.com/galilai-group/stable-worldmodel · https://github.com/lucas-maes/le-wm
