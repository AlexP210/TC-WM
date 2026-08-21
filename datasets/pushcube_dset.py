"""PushCube-v1 / PushCube-v1.1 trajectories.

Only the flat-vector layout is here; everything about how a ManiSkill recording is read lives in
maniskill_dset.py.
"""
from .maniskill_dset import ManiSkillTrajDataset, load_maniskill_slice_train_val

# These MUST stay in sync with env/pushcube/pushcube_wrapper.py, which slices the same vectors
# back apart by offset.
PROPRIO_KEYS = ["obs/agent/qpos", "obs/agent/qvel", "obs/extra/tcp_pose"]
STATE_KEYS = [
    "env_states/articulations/panda",
    "env_states/actors/cube",
    "env_states/actors/goal_region",
    "env_states/actors/table-workspace",
]


class PushBlockDataset(ManiSkillTrajDataset):
    """A PushCube recording: the panda, the cube, its goal region and the table (70-wide state)."""

    proprio_keys = PROPRIO_KEYS
    state_keys = STATE_KEYS


def load_pushcube_slice_train_val(
    transform,
    data_path,
    n_rollout=None,
    normalize_action=False,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    camera=None,
):
    return load_maniskill_slice_train_val(
        PushBlockDataset,
        transform=transform,
        data_path=data_path,
        n_rollout=n_rollout,
        normalize_action=normalize_action,
        split_ratio=split_ratio,
        num_hist=num_hist,
        num_pred=num_pred,
        frameskip=frameskip,
        camera=camera,
    )
