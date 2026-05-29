import torch
import decord
import pickle
import numpy as np
import os
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Callable, Optional
from .traj_dset import TrajDataset, TrajSlicerDataset
from typing import Optional, Callable, Any
decord.bridge.set_bridge("torch")

# robomimic dataset precomputed statistics (computed from actual data with velocities)
ACTION_MEAN = torch.tensor([0.14458055794239044, 0.20117317140102386])  # 2D actions [dx, dy]
ACTION_STD = torch.tensor([0.3066084086894989, 0.5769815444946289])
STATE_MEAN = torch.tensor([-0.029855675995349884, 0.5357389450073242, 0.02471051551401615, 0.006146272178739309, 0.018406163901090622, 0.04309431090950966, 0.11412229388952255])  # 7D states [agent_x, agent_y, block_x, block_y, angle, vel_x, vel_y]
STATE_STD = torch.tensor([0.1633525937795639, 0.3669397532939911, 0.014770451001822948, 0.03916474059224129, 0.047178756445646286, 0.1489015817642212, 0.3091855049133301])
PROPRIO_MEAN = torch.tensor([-0.029855675995349884, 0.5357389450073242, 0.02471051551401615, 0.006146272178739309, 0.018406163901090622, 0.04309431090950966, 0.11412229388952255])  # 7D proprio
PROPRIO_STD = torch.tensor([0.1633525937795639, 0.3669397532939911, 0.014770451001822948, 0.03916474059224129, 0.047178756445646286, 0.1489015817642212, 0.3091855049133301])

class RobomimicDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,  # Required parameter, no default
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        relative=True,
        action_scale=1.0,
        with_velocity: bool = True, # agent's velocity
    ):  
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalize_action = normalize_action
        self.states = torch.load(self.data_path / "states.pth")
        self.states = self.states.float()
        if relative:
            self.actions = torch.load(self.data_path / "rel_actions.pth")
        else:
            self.actions = torch.load(self.data_path / "abs_actions.pth")
        self.actions = self.actions.float()
        self.actions = self.actions / action_scale  # scaled back up in env

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)
        
        # load shapes, assume all shapes are 'T' if file not found
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, 'rb') as f:
                shapes = pickle.load(f)
                self.shapes = shapes
        else:
            self.shapes = ['T'] * len(self.states)

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        
        # Try to load separate proprios file (new format), fallback to old method
        try:
            self.proprios = torch.load(self.data_path / "proprios.pth")
            self.proprios = self.proprios[:n].float()
            print(f"Loaded separate proprios.pth ({self.proprios.shape[-1]}D proprios)")
        except FileNotFoundError:
            print(f"No proprios.pth found, using legacy method")
            self.proprios = self.states.clone()  # Legacy: states are proprio
            
        # load velocities and update states and proprios (legacy behavior)
        self.with_velocity = with_velocity
        if with_velocity and not (self.data_path / "proprios.pth").exists():
            # Only do this for legacy datasets without separate proprios.pth
            self.velocities = torch.load(self.data_path / "velocities.pth")
            self.velocities = self.velocities[:n].float()
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        elif with_velocity:
            # For new format, just load velocities separately (they're already in proprios)
            self.velocities = torch.load(self.data_path / "velocities.pth")
            self.velocities = self.velocities[:n].float()
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            # 使用预计算的统计信息或从数据中计算
            try:
                self.action_mean = torch.load(self.data_path / "action_mean.pth")
                self.action_std = torch.load(self.data_path / "action_std.pth")
                self.state_mean = torch.load(self.data_path / "state_mean.pth")
                self.state_std = torch.load(self.data_path / "state_std.pth")
                self.proprio_mean = torch.load(self.data_path / "proprio_mean.pth")
                self.proprio_std = torch.load(self.data_path / "proprio_std.pth")
            except FileNotFoundError:
                raise FileNotFoundError("Statistics file not found in robomimic dataset!")
                print("Using default statistics for robomimic dataset")
                self.action_mean = ACTION_MEAN[:self.action_dim]
                self.action_std = ACTION_STD[:self.action_dim]
                self.state_mean = STATE_MEAN[:self.state_dim]
                self.state_std = STATE_STD[:self.state_dim]
                self.proprio_mean = PROPRIO_MEAN[:self.proprio_dim]
                self.proprio_std = PROPRIO_STD[:self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        vid_dir = self.data_path / "obses"
        reader = VideoReader(str(vid_dir / f"episode_{idx:05d}.mp4"), num_threads=1)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]

        image = reader.get_batch(frames)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.asnumpy())
        image = image.float() / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {'shape': shape}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_robomimic_slice_train_val(
    transform,
    data_path,  # Required parameter, no default
    n_rollout=50,
    normalize_action=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
):
    # 检查是否有 train/val 分割
    train_path = data_path + "/train"
    val_path = data_path + "/val"
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        # 如果有分割，使用分割的数据
        train_dset = RobomimicDataset(
            data_path=train_path,
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            with_velocity=with_velocity,
        )
        val_dset = RobomimicDataset(
            data_path=val_path,
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            with_velocity=with_velocity,
        )
    else:
        # 如果没有分割，创建一个数据集并手动分割
        print(f"No train/val split found, using single dataset at {data_path}")
        full_dset = RobomimicDataset(
            data_path=data_path,
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            with_velocity=with_velocity,
        )
        
        # 手动分割数据
        total_samples = len(full_dset)
        train_size = int(total_samples * split_ratio)
        val_size = total_samples - train_size
        
        print(f"Total samples: {total_samples}, Train: {train_size}, Val: {val_size}")
        
        # 创建训练和验证数据集
        from torch.utils.data import Subset
        
        class RobomimicSubset(Subset):
            def __init__(self, dataset, indices):
                super().__init__(dataset, indices)
                self.dataset = dataset
                self.indices = indices
            
            def get_seq_length(self, idx):
                return self.dataset.get_seq_length(self.indices[idx])
            
            def get_all_actions(self):
                return self.dataset.get_all_actions()
            @property
            def states(self):
                return self.dataset.states
            @property
            def proprios(self):
                return self.dataset.proprios
            @property
            def actions(self):
                return self.dataset.actions
            @property
            def shapes(self):
                return self.dataset.shapes
            @property
            def action_dim(self):
                return self.dataset.action_dim
            
            @property
            def state_dim(self):
                return self.dataset.state_dim
            
            @property
            def proprio_dim(self):
                return self.dataset.proprio_dim
            
            @property
            def action_mean(self):
                return self.dataset.action_mean
            
            @property
            def action_std(self):
                return self.dataset.action_std
            
            @property
            def state_mean(self):
                return self.dataset.state_mean
            
            @property
            def state_std(self):
                return self.dataset.state_std
            
            @property
            def proprio_mean(self):
                return self.dataset.proprio_mean
            
            @property
            def proprio_std(self):
                return self.dataset.proprio_std
            
            @property
            def transform(self):
                return self.dataset.transform
        
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_samples))
        
        train_dset = RobomimicSubset(full_dset, train_indices)
        val_dset = RobomimicSubset(full_dset, val_indices)

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    return datasets, traj_dset 