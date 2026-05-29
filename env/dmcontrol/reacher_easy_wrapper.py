"""
dm_control reacher-easy wrapper for goal-conditioned CEM planning.

Implements the env interface required by plan.py:
  - rollout(seed, init_state, actions) -> (obses, states)
  - prepare(seed, init_state) -> (obs, state)
  - sample_random_init_goal_states(seeds) -> (init_states, goal_states)
  - eval_state(goal_state, cur_state) -> dict with 'success', 'state_dist'
  - update_env(env_info) -> None (no-op)

State layout (4D): [qpos(2), qvel(2)]
  [0:2] = joint positions (shoulder, wrist)
  [2:4] = joint velocities

Proprio layout (4D): [position(2), velocity(2)] — no to_target.
Target position is NOT stored in state; kept in env physics from reset.
eval_state uses FK-based fingertip distance (independent of target).
"""

import os

import gym
from gym import spaces
import numpy as np
from dm_control import suite
from torchvision import transforms

# Force offscreen rendering
os.environ.setdefault("MUJOCO_GL", "egl")

IMG_SIZE = 224
TRANSFORM = transforms.Resize((IMG_SIZE, IMG_SIZE))

# Success threshold: fingertip within this distance of target
SUCCESS_THRESHOLD = 0.05


ACTION_REPEAT = 2  # Must match TDMPC2/NEWT data collection (hardcoded 2)


class ReacherEasyWrapper(gym.Env):
    def __init__(self, **kwargs):
        self._env = suite.load("reacher", "easy", task_kwargs={"random": 42})
        self.action_dim = self._env.action_spec().shape[0]  # 2
        self.action_repeat = ACTION_REPEAT
        self.transform = TRANSFORM
        # gym requires action_space / observation_space
        act_spec = self._env.action_spec()
        self.action_space = spaces.Box(
            low=act_spec.minimum, high=act_spec.maximum, dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "visual": spaces.Box(0, 255, shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.float32),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
        })

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _get_target_xy(self):
        """Get current target xy position (absolute)."""
        return self._env.physics.named.data.geom_xpos["target"][:2].copy()

    def _set_target_xy(self, target_xy):
        """Restore target xy position in the model."""
        self._env.physics.named.model.geom_pos["target", :2] = target_xy

    def _get_finger_xy(self):
        """Get current fingertip xy position."""
        return self._env.physics.named.data.geom_xpos["finger"][:2].copy()

    def _get_obs(self):
        """Return obs dict with 'visual' (H,W,C) and 'proprio'.

        Proprio layout: position(2) + velocity(2) = 4D (no to_target).
        """
        pixels = self._env.physics.render(height=IMG_SIZE, width=IMG_SIZE, camera_id=0)
        pos = self._env.physics.named.data.qpos[["shoulder", "wrist"]].copy()
        vel = self._env.physics.named.data.qvel[["shoulder", "wrist"]].copy()
        proprio = np.concatenate([pos, vel]).astype(np.float32)
        return {
            "visual": pixels.astype(np.float32),  # (H,W,C) float [0, 255]
            "proprio": proprio,
        }

    def _get_state(self):
        """Return state: [qpos(2), qvel(2)] = 4D. No to_target."""
        return np.concatenate([
            self._env.physics.data.qpos[:].copy(),  # (2,)
            self._env.physics.data.qvel[:].copy(),  # (2,)
        ]).astype(np.float32)

    def _set_state(self, state):
        """Restore state from [qpos(2), qvel(2)] = 4D. Target position unchanged."""
        nq = self._env.physics.model.nq  # 2
        nv = self._env.physics.model.nv  # 2
        self._env.physics.data.qpos[:] = state[:nq]
        self._env.physics.data.qvel[:] = state[nq:nq + nv]
        self._env.physics.after_reset()


    # ------------------------------------------------------------------
    # planning interface
    # ------------------------------------------------------------------
    def seed(self, s):
        pass  # dm_control seed set at load time

    def prepare(self, seed, init_state):
        """Reset env to init_state, return (obs, state).

        Does NOT call _env.reset() to avoid randomizing the target position.
        Target is restored from the state vector.
        """
        # Minimal reset: clear physics without randomizing target
        self._env.physics.reset()
        self._set_state(init_state)
        obs = self._get_obs()
        state = self._get_state()
        return obs, state

    def rollout(self, seed, init_state, actions):
        """Execute actions from init_state, return (obses, states).

        Each action is repeated action_repeat times to match TDMPC2 data collection.
        """
        self._env.physics.reset()
        self._set_state(init_state)

        traj_obs = [self._get_obs()]
        traj_states = [self._get_state()]

        for t in range(actions.shape[0]):
            act = actions[t].clip(
                self._env.action_spec().minimum,
                self._env.action_spec().maximum,
            )
            for _ in range(self.action_repeat):
                self._env.step(act)
            traj_obs.append(self._get_obs())
            traj_states.append(self._get_state())

        obses = {
            k: np.stack([o[k] for o in traj_obs])
            for k in traj_obs[0].keys()
        }
        states = np.stack(traj_states)
        return obses, states, None

    def sample_random_init_goal_states(self, seed):
        """Sample random (init, goal) state pair: [qpos(2), qvel(2)] = 4D."""
        rng = np.random.RandomState(seed)

        # Reset once to set target position (kept for visual consistency)
        self._env.reset()

        # Random init
        self._env.physics.data.qpos[:] = rng.uniform(-np.pi, np.pi, size=self._env.physics.model.nq)
        self._env.physics.data.qvel[:] = np.zeros(self._env.physics.model.nv)
        self._env.physics.after_reset()
        init_state = self._get_state()

        # Random goal (same target, different joint positions)
        self._env.physics.data.qpos[:] = rng.uniform(-np.pi, np.pi, size=self._env.physics.model.nq)
        self._env.physics.data.qvel[:] = np.zeros(self._env.physics.model.nv)
        self._env.physics.after_reset()
        goal_state = self._get_state()

        return init_state, goal_state

    def eval_state(self, goal_state, cur_state):
        """Evaluate if current state reached goal via fingertip distance.

        Sets each state in physics to compute fingertip position via FK.
        """
        self._env.physics.reset()
        self._set_state(goal_state)
        goal_finger = self._get_finger_xy()

        self._env.physics.reset()
        self._set_state(cur_state)
        cur_finger = self._get_finger_xy()

        dist = float(np.linalg.norm(goal_finger - cur_finger))
        return {
            "success": dist < SUCCESS_THRESHOLD,
            "state_dist": dist,
        }

    def update_env(self, env_info):
        """No-op for dm_control."""
        pass

    def step(self, action):
        reward = 0.0
        for _ in range(self.action_repeat):
            ts = self._env.step(action)
            reward += ts.reward or 0.0
        obs = self._get_obs()
        done = ts.last()
        return obs, reward, done, {}

    def reset(self):
        self._env.reset()
        return self._get_obs()
