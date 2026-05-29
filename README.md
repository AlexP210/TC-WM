# Back to Parsimonious Latents: Learning Task-Centric World Models from Visual Foundations

Turning frozen visual foundation embeddings into a compact, task-centric latent space for reward-free offline planning and control.

[Minghao Fu](https://minghaofu.github.io/) &middot; [Fan Feng](https://scholar.google.com/citations?user=rYKQ3i8AAAAJ) &middot; [Nicklas Hansen](https://www.nicklashansen.com/) &middot; [Biwei Huang](https://biweihuang.com/) &nbsp;|&nbsp; UC San Diego

[![arXiv](https://img.shields.io/badge/arXiv-2605.25620-b31b1b)](https://arxiv.org/abs/2605.25620)
[![Project](https://img.shields.io/badge/Project-Page-2563eb)](https://minghaofu.com/tc-wm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What

TC-WM treats a pretrained visual backbone (e.g. DINOv2) as a **semantic scaffold**, not the final state space. A **linear projection** compresses its embedding into a compact latent; a designated subspace is **aligned with proprioception** via InfoNCE; a **ViT** predicts latent dynamics; a **linear decoder** reconstructs the embedding to prevent collapse. The task-centric block is identifiable up to a simple transformation.

Empirically, TC-WM enables zero-shot **test-time planning** across nine offline visual-control tasks — Maze, Wall, Push-T, Lift, Can, Square, Reacher, Cheetah, Hopper — beating DINO-WM on every LDP task and matching strong model-based baselines.

## Install

```bash
conda env create -f environment.yml
conda activate tcwm
bash install_mujoco.sh             # MuJoCo 210 + LD_LIBRARY_PATH setup
pip install d4rl pymunk==6.* shapely scikit-image pygame   # env extras
```

Set the data root once:

```bash
export TCWM_DATA_ROOT=/path/to/data       # configs read this via ${oc.env:TCWM_DATA_ROOT,./data}
```

## Train

```bash
# Single task
python train.py --config-name=train_tcwm env=wall

# All TC-WM tasks (one GPU each, picks free GPUs automatically)
bash run_tcwm.sh
```

Configs live in `conf/` (Hydra). Key knobs: `env=`, `encoder=` (dino / vjepa / dinov3), `projected_dim`, `alignment_dim`, `training.epochs`. Offline run by default (`WANDB_MODE=offline`).

## Plan

```bash
python plan.py --config-name=plan_wall \
  ckpt_base_path=$TCWM_DATA_ROOT/checkpoints/wall_proj256 \
  n_evals=50
```

Available planners: CEM (`conf/planner/cem.yaml`), LDP (`conf/planner/ldp.yaml`), GD. Rollout videos auto-saved as `plan{batch}_{trial}_{success|failure}.mp4` in the Hydra run dir.

## Rollout (qualitative)

```bash
python rollout.py --config-name=train_tcwm \
  env=wall resume_folder=$TCWM_DATA_ROOT/checkpoints/wall_proj256 \
  +traj_idx=0 +output_dir=rollouts/wall_idx0 +fps=8
```

Saves three GIFs per trajectory: `{env}_orig.gif`, `{env}_recon.gif`, `{env}_pred.gif` &mdash; original observation, autoencoder reconstruction, open-loop TC-WM prediction.

## Repository layout

```
TC-WM/
├── train.py · plan.py · rollout.py · utils.py · preprocessor.py · custom_resolvers.py
├── models/         # VWorldModel + encoder · projector · predictor · decoder
├── env/            # Gym wrappers for Maze, Wall, Push-T, Robomimic, DMC
├── datasets/       # Trajectory loaders
├── planning/       # CEM, LDP, MPC, evaluator
├── conf/           # Hydra configs (env / encoder / method / planner / ...)
├── metrics/  distributed_fn/  gpu_utils/
└── assets/
```

## Checkpoints

Pretrained checkpoints will be released at [`MinghaoFu/TC-WM-checkpoints`](https://huggingface.co/MinghaoFu/TC-WM-checkpoints) on HuggingFace Hub. Each task ships with a single best-seed checkpoint; load with `resume_folder=<path>`.

## Citation

```bibtex
@article{fu2026tcwm,
  title   = {Back to Parsimonious Latents: Learning Task-Centric World Models from Visual Foundations},
  author  = {Fu, Minghao and Feng, Fan and Hansen, Nicklas and Huang, Biwei},
  journal = {arXiv preprint arXiv:2605.25620},
  year    = {2026}
}
```

## Contact

`m9fu [at] ucsd [dot] edu`
