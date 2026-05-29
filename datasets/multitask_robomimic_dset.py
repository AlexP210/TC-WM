"""
Multi-task robomimic dataset.

Each sub-dataset (lift, can, square, ...) is assigned a task_id (0-indexed).
The combined dataset maps global episode indices to (sub_dataset, local_idx).
TrajSlicerDataset is extended to pass task_id through to the training loop.

Returns per sample: (obs, act, state, task_id)
  obs:     dict with "visual" (T, C, H, W) and "proprio" (T, proprio_dim)
  act:     (T, action_dim * frameskip)
  state:   (T, state_dim)
  task_id: scalar LongTensor
"""

import bisect
import json
import os
import pickle
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from einops import rearrange

from .robomimic_dset import RobomimicDataset
from .traj_dset import TrajDataset


# ---------------------------------------------------------------------------
# Per-dataset loader functions — each absorbs irrelevant kwargs via **_
# Most datasets share _make_robomimic; add exceptions below for different formats.
# ---------------------------------------------------------------------------

def _make_robomimic(data_path, n_rollout, transform, normalize_action, with_velocity=True, **_):
    return RobomimicDataset(data_path=data_path, n_rollout=n_rollout,
                            transform=transform, normalize_action=normalize_action,
                            with_velocity=with_velocity)

def _make_point_maze(data_path, n_rollout, transform, normalize_action, **_):
    from .point_maze_dset import PointMazeDataset
    return PointMazeDataset(data_path=data_path, n_rollout=n_rollout,
                            transform=transform, normalize_action=normalize_action)

def _make_wall(data_path, n_rollout, transform, normalize_action, **_):
    from .wall_dset import WallDataset
    return WallDataset(data_path=data_path, n_rollout=n_rollout,
                       transform=transform, normalize_action=normalize_action)

def _make_pusht(data_path, n_rollout, transform, normalize_action, with_velocity=True, **_):
    from .pusht_dset import PushTDataset
    return PushTDataset(data_path=data_path, n_rollout=n_rollout,
                        transform=transform, normalize_action=normalize_action,
                        with_velocity=with_velocity)

# Registry: task_name → loader fn
# To add a new task:
#   1. Write a _make_* fn above (or reuse an existing one if format matches)
#   2. Add one line here
_TASK_REGISTRY = {
    # robomimic manipulation tasks (all share same data format)
    "lift":       _make_robomimic,
    "can":        _make_robomimic,
    "square":     _make_robomimic,
    "transport":  _make_robomimic,
    "tool_hang":  _make_robomimic,
    # navigation / manipulation tasks with different data formats
    "point_maze": _make_point_maze,
    "wall":       _make_wall,
    "pusht":      _make_pusht,
}


# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------

class MultiTaskRobomimicDataset(TrajDataset):
    """
    Concatenates multiple RobomimicDatasets. Each is assigned a task_id int.
    Proprios are zero-padded to max_proprio_dim across all sub-datasets.
    """

    def __init__(
        self,
        sub_datasets: List[RobomimicDataset],
        task_names: List[str],
        max_proprio_dim: Optional[int] = None,
        max_state_dim: Optional[int] = None,
    ):
        assert len(sub_datasets) == len(task_names)
        self.sub_datasets = sub_datasets
        self.task_names = task_names
        self.task_name_to_id = {name: i for i, name in enumerate(task_names)}

        # cumulative episode counts for O(log n) index mapping
        self._cumulative = [0]
        for ds in sub_datasets:
            self._cumulative.append(self._cumulative[-1] + len(ds))

        # unified dims (pad to max across tasks)
        self.max_proprio_dim = max_proprio_dim or max(ds.proprio_dim for ds in sub_datasets)
        self.max_state_dim = max_state_dim or max(ds.state_dim for ds in sub_datasets)
        self.max_action_dim = max(ds.action_dim for ds in sub_datasets)

        # expose unified dims for train_uwm.py compatibility
        self.proprio_dim = self.max_proprio_dim
        self.state_dim = self.max_state_dim
        self.action_dim = self.max_action_dim
        self.num_tasks = len(sub_datasets)

    def _resolve(self, global_idx: int):
        """Map global episode index → (task_id, sub_dataset, local_idx)."""
        task_id = bisect.bisect_right(self._cumulative, global_idx) - 1
        task_id = min(task_id, len(self.sub_datasets) - 1)
        local_idx = global_idx - self._cumulative[task_id]
        return task_id, self.sub_datasets[task_id], local_idx

    def get_seq_length(self, global_idx: int) -> int:
        _, ds, local_idx = self._resolve(global_idx)
        return ds.get_seq_length(local_idx)

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(self, global_idx: int):
        task_id, ds, local_idx = self._resolve(global_idx)
        obs, act, state, extras = ds[local_idx]

        # zero-pad proprio if needed
        if obs["proprio"].shape[-1] < self.max_proprio_dim:
            pad = self.max_proprio_dim - obs["proprio"].shape[-1]
            obs["proprio"] = torch.nn.functional.pad(obs["proprio"], (0, pad))

        # zero-pad action if needed (transport has 14D, others 7D)
        if act.shape[-1] < self.max_action_dim:
            pad = self.max_action_dim - act.shape[-1]
            act = torch.nn.functional.pad(act, (0, pad))

        # zero-pad state if needed
        if state.shape[-1] < self.max_state_dim:
            pad = self.max_state_dim - state.shape[-1]
            state = torch.nn.functional.pad(state, (0, pad))

        extras["task_id"] = task_id
        return obs, act, state, extras


# ---------------------------------------------------------------------------
# Slicer (extends TrajSlicerDataset to pass task_id through)
# ---------------------------------------------------------------------------

class MultiTaskTrajSlicerDataset(TrajDataset):
    """
    Task-balanced slice dataset.

    Each __getitem__ call does two-step iid sampling:
      1. Sample task_id ~ Uniform({0, ..., n_tasks-1})
      2. Sample a random slice from that task's slice pool

    This gives each task equal expected representation per batch regardless of
    episode length or dataset-size differences (unlike flat-concat which biases
    toward tasks with longer episodes / more slices).

    Epoch length = min_task_slices * n_tasks  (or steps_per_epoch if provided).
    """

    def __init__(
        self,
        dataset: MultiTaskRobomimicDataset,
        num_frames: int,
        frameskip: int = 1,
        steps_per_epoch: Optional[int] = None,
    ):
        self.dataset = dataset
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.n_tasks = len(dataset.sub_datasets)

        # Build per-task slice lists; store global indices so __getitem__ can
        # delegate to MultiTaskRobomimicDataset (which handles zero-padding).
        window = num_frames * frameskip
        self.per_task_slices: List[np.ndarray] = []
        for task_id, sub_ds in enumerate(dataset.sub_datasets):
            task_start = dataset._cumulative[task_id]
            slices = []
            for local_idx in range(len(sub_ds)):
                T = sub_ds.get_seq_length(local_idx)
                if T - window + 1 < 1:
                    print(
                        f"Ignored short sequence #{local_idx} in task "
                        f"{dataset.task_names[task_id]}: len={T}"
                    )
                else:
                    global_idx = task_start + local_idx
                    for start in range(T - window + 1):
                        slices.append((global_idx, start, start + window))
            self.per_task_slices.append(np.array(slices, dtype=np.int64))

        task_sizes = [len(s) for s in self.per_task_slices]
        min_size = min(task_sizes)
        for name, size in zip(dataset.task_names, task_sizes):
            print(f"  [{name}] slices={size}")
        print(
            f"  Task-balanced epoch: min={min_size} × {self.n_tasks} tasks"
            f" = {min_size * self.n_tasks} items"
            f" ({min_size * self.n_tasks // 16} batches @ bs=16)"
        )

        self._len = steps_per_epoch if steps_per_epoch is not None else min_size * self.n_tasks

        # propagate dims for train_uwm.py compatibility
        self.proprio_dim = dataset.proprio_dim
        self.state_dim = dataset.state_dim
        self.action_dim = dataset.action_dim * frameskip
        self.num_tasks = dataset.num_tasks
        self.task_names = dataset.task_names

    def get_seq_length(self, idx: int) -> int:
        return self.num_frames

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        # Step 1: sample task uniformly (idx is ignored — DataLoader provides it
        # but we do our own balanced sampling)
        task_id = int(np.random.randint(0, self.n_tasks))
        # Step 2: sample a random slice from that task's pool
        slices = self.per_task_slices[task_id]
        s = slices[int(np.random.randint(len(slices)))]
        global_idx, start, end = int(s[0]), int(s[1]), int(s[2])

        # Load via MultiTaskRobomimicDataset — handles zero-padding for all dims
        obs, act, state, extras = self.dataset[global_idx]

        for k, v in obs.items():
            obs[k] = v[start:end:self.frameskip]
        state = state[start:end:self.frameskip]
        act = act[start:end]
        act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)

        obs["task_id"] = torch.tensor(extras["task_id"], dtype=torch.long)
        return obs, act, state


# ---------------------------------------------------------------------------
# Factory function (Hydra _target_)
# ---------------------------------------------------------------------------

def load_multitask_robomimic_slice_train_val(
    transform: Callable,
    tasks: List[dict],           # [{name, data_path, n_rollout, ...}, ...]
    normalize_action: bool = True,
    split_ratio: float = 0.9,
    num_hist: int = 0,
    num_pred: int = 0,
    frameskip: int = 1,
    with_velocity: bool = True,
):
    """
    Load multiple robomimic datasets and combine into a MultiTaskRobomimicDataset.

    Args:
        tasks: list of task configs, each with:
            name       (str)  task name, e.g. "lift"
            data_path  (str)  path to converted dataset directory
            n_rollout  (int, optional) cap on number of episodes
    """
    num_frames = num_hist + num_pred

    train_sub, val_sub, task_names = [], [], []

    for task_cfg in tasks:
        name      = task_cfg["name"]
        data_path = task_cfg["data_path"]
        n_rollout = task_cfg.get("n_rollout", None)

        if name not in _TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{name}'. Add it to _TASK_REGISTRY "
                f"in datasets/multitask_robomimic_dset.py"
            )

        loader_fn = _TASK_REGISTRY[name]
        common_kw = dict(n_rollout=n_rollout, transform=transform,
                         normalize_action=normalize_action, with_velocity=with_velocity)

        train_path = os.path.join(data_path, "train")
        val_path   = os.path.join(data_path, "val")

        if os.path.exists(train_path) and os.path.exists(val_path):
            tr_ds = loader_fn(data_path=train_path, **common_kw)
            va_ds = loader_fn(data_path=val_path,   **common_kw)
        else:
            print(f"[{name}] No train/val split found, splitting manually from {data_path}")
            full_ds = loader_fn(data_path=data_path, **common_kw)
            n_train = int(len(full_ds) * split_ratio)
            tr_ds   = _SubsetWrapper(full_ds, list(range(n_train)))
            va_ds   = _SubsetWrapper(full_ds, list(range(n_train, len(full_ds))))

        train_sub.append(tr_ds)
        val_sub.append(va_ds)
        task_names.append(name)
        print(f"  [{name}] train={len(tr_ds)}, val={len(va_ds)}, proprio={tr_ds.proprio_dim}D")

    train_combined = MultiTaskRobomimicDataset(train_sub, task_names)
    val_combined = MultiTaskRobomimicDataset(val_sub, task_names)

    print(f"MultiTask: {len(task_names)} tasks, max_proprio={train_combined.max_proprio_dim}D")

    train_slices = MultiTaskTrajSlicerDataset(train_combined, num_frames, frameskip)
    val_slices = MultiTaskTrajSlicerDataset(val_combined, num_frames, frameskip)

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dsets = {"train": train_combined, "valid": val_combined}
    return datasets, traj_dsets


# ---------------------------------------------------------------------------
# Minimal Subset wrapper that proxies RobomimicDataset attributes
# ---------------------------------------------------------------------------

class _SubsetWrapper:
    """Thin wrapper around a RobomimicDataset subset (no full TrajSubset overhead)."""

    def __init__(self, dataset: RobomimicDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def get_seq_length(self, idx: int) -> int:
        return self.dataset.get_seq_length(self.indices[idx])

    # proxy all dataset attributes
    def __getattr__(self, name):
        return getattr(self.dataset, name)
