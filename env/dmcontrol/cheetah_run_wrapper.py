"""
dm_control cheetah-run wrapper for goal-conditioned CEM planning.

Implements the env interface required by plan.py:
  - rollout(seed, init_state, actions) -> (obses, states)
  - prepare(seed, init_state) -> (obs, state)
  - sample_random_init_goal_states(seeds) -> (init_states, goal_states)
  - eval_state(goal_state, cur_state) -> dict with 'success', 'state_dist'
  - update_env(env_info) -> None (no-op)

State layout (TDMPC2 format, 17D): [position(8), velocity(9)]
  position = qpos[1:] (rootx excluded), velocity = qvel
Proprio layout (17D): position(8) + velocity(9) — matches TDMPC2 format.
"""

import os

import gym
from gym import spaces
import numpy as np
from dm_control import suite
from torchvision import transforms

os.environ.setdefault("MUJOCO_GL", "egl")

IMG_SIZE = 224
TRANSFORM = transforms.Resize((IMG_SIZE, IMG_SIZE))

# Success threshold: position (qpos[1:]) L2 distance
SUCCESS_THRESHOLD = 0.15
ACTION_REPEAT = 2  # Must match TDMPC2/NEWT data collection (hardcoded 2)


class CheetahRunWrapper(gym.Env):
    def __init__(self, **kwargs):
        self._env = suite.load("cheetah", "run", task_kwargs={"random": 42})
        self.action_dim = self._env.action_spec().shape[0]  # 6
        self.action_repeat = ACTION_REPEAT
        self.transform = TRANSFORM
        act_spec = self._env.action_spec()
        self.action_space = spaces.Box(
            low=act_spec.minimum, high=act_spec.maximum, dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "visual": spaces.Box(0, 255, shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.float32),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32),
        })

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _get_obs(self):
        pixels = self._env.physics.render(height=IMG_SIZE, width=IMG_SIZE, camera_id=0)
        obs = self._env.task.get_observation(self._env.physics)
        position = obs["position"]  # (8,)
        velocity = obs["velocity"]  # (9,)
        proprio = np.concatenate([position, velocity]).astype(np.float32)
        return {
            "visual": pixels.astype(np.float32),
            "proprio": proprio,
        }

    def _get_state(self):
        """Return state in TDMPC2 format: [position(8), velocity(9)] = 17D.

        position = qpos[1:] (rootx excluded), velocity = qvel.
        """
        obs = self._env.task.get_observation(self._env.physics)
        return np.concatenate([
            obs["position"],  # qpos[1:], 8D
            obs["velocity"],  # qvel, 9D
        ]).astype(np.float32)

    def _set_state(self, state):
        """Set state from TDMPC2 format: [position(8), velocity(9)].

        position = qpos[1:], so we set qpos[0] (rootx) to 0.
        """
        nq = self._env.physics.model.nq  # 9
        nv = self._env.physics.model.nv  # 9
        if len(state) == nq + nv:
            # Raw qpos+qvel format (18D)
            self._env.physics.data.qpos[:] = state[:nq]
            self._env.physics.data.qvel[:] = state[nq:nq + nv]
        else:
            # TDMPC2 format (17D): position(8) + velocity(9)
            self._env.physics.data.qpos[0] = 0.0  # rootx not in observation
            self._env.physics.data.qpos[1:] = state[:nq - 1]
            self._env.physics.data.qvel[:] = state[nq - 1:nq - 1 + nv]
        self._env.physics.after_reset()

    # ------------------------------------------------------------------
    # planning interface
    # ------------------------------------------------------------------
    def seed(self, s):
        pass

    def prepare(self, seed, init_state):
        self._env.physics.reset()
        self._set_state(init_state)
        return self._get_obs(), self._get_state()

    def rollout(self, seed, init_state, actions):
        """Each action is repeated action_repeat times to match TDMPC2 data collection."""
        self._env.physics.reset()
        self._set_state(init_state)

        traj_obs = [self._get_obs()]
        traj_states = [self._get_state()]
        traj_rewards = []

        for t in range(actions.shape[0]):
            act = actions[t].clip(
                self._env.action_spec().minimum,
                self._env.action_spec().maximum,
            )
            step_reward = 0.0
            for _ in range(self.action_repeat):
                ts = self._env.step(act)
                step_reward += ts.reward or 0.0
            traj_obs.append(self._get_obs())
            traj_states.append(self._get_state())
            traj_rewards.append(step_reward)

        obses = {
            k: np.stack([o[k] for o in traj_obs])
            for k in traj_obs[0].keys()
        }
        states = np.stack(traj_states)
        rewards = np.array(traj_rewards)
        return obses, states, rewards

    def rollout_reward_only(self, init_state, actions):
        """Fast rollout without rendering — only returns cumulative reward."""
        self._env.physics.reset()
        self._set_state(init_state)
        total_reward = 0.0
        for t in range(actions.shape[0]):
            act = actions[t].clip(
                self._env.action_spec().minimum,
                self._env.action_spec().maximum,
            )
            for _ in range(self.action_repeat):
                ts = self._env.step(act)
                total_reward += ts.reward or 0.0
        return total_reward

    def sample_random_init_goal_states(self, seed):
        """Sample random states in TDMPC2 format: [position(8), velocity(9)]."""
        rng = np.random.RandomState(seed)

        # Random init: larger perturbation for meaningful goal diversity
        self._env.reset()
        self._env.physics.data.qpos[1:] += rng.uniform(-0.5, 0.5, size=self._env.physics.model.nq - 1)
        self._env.physics.data.qvel[:] += rng.uniform(-0.5, 0.5, size=self._env.physics.model.nv)
        self._env.physics.after_reset()
        init_state = self._get_state()

        # Random goal
        self._env.reset()
        self._env.physics.data.qpos[1:] += rng.uniform(-0.5, 0.5, size=self._env.physics.model.nq - 1)
        self._env.physics.data.qvel[:] += rng.uniform(-0.5, 0.5, size=self._env.physics.model.nv)
        self._env.physics.after_reset()
        goal_state = self._get_state()

        return init_state, goal_state

    def eval_state(self, goal_state, cur_state):
        """Evaluate using position (qpos[1:]) from TDMPC2 state format."""
        nq_minus_1 = self._env.physics.model.nq - 1  # 8 (position dims)
        goal_pos = goal_state[:nq_minus_1]
        cur_pos = cur_state[:nq_minus_1]
        dist = float(np.linalg.norm(goal_pos - cur_pos))
        return {
            "success": dist < SUCCESS_THRESHOLD,
            "state_dist": dist,
        }

    def update_env(self, env_info):
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
