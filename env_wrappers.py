import gymnasium as gym
import config
import sys
import ale_py
import numpy as np
import cv2
from gymnasium.spaces import Box
from collections import deque


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1) # FIRE
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2) # RIGHT
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about them.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame and max over the last 2 frames.
    """
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        terminated = False
        truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip rewards to {+1, 0, -1}.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    """
    Warp frames to 84x84 as done in the Nature paper and later work.
    """
    def __init__(self, env, width=84, height=84):
        super(WarpFrame, self).__init__(env)
        self.width = width
        self.height = height
        self.observation_space = Box(
            low=0, high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8
        )

    def observation(self, frame):
        """Convert to grayscale and resize observation."""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    """
    Stack k last frames.
    Returns lazy array, which is much more memory efficient.
    """
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(
            low=0, high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=2)


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Normalize observations to be between 0 and 1.
    """
    def __init__(self, env):
        super(ScaledFloatFrame, self).__init__(env)
        self.observation_space = Box(
            low=0, high=1,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        # Careful! This undoes the memory optimization
        return np.array(observation).astype(np.float32) / 255.0


def make_env(env_name=config.ENV_NAME, render_mode=config.RENDER_MODE):
    """
    Create an Atari Ice Hockey environment with appropriate wrappers.
    
    Args:
        env_name (str): Name of the Atari environment to create
        render_mode (str): Rendering mode, None for no rendering or 'human' for visualization
    
    Returns:
        gym.Env: A wrapped gymnasium environment ready for DQN training
    """
    try:
        # Try to create the base environment with lowest difficulty (0)
        env = gym.make(env_name, render_mode=render_mode, difficulty=0, mode=0)
        
        # Apply wrappers in standard order
        
        # 1. NoopResetEnv: Start episodes with random number of no-ops
        env = NoopResetEnv(env, noop_max=config.NOOP_MAX)
        
        # 2. MaxAndSkipEnv: Skip frames but retain max values to capture fast movements
        env = MaxAndSkipEnv(env, skip=config.FRAME_SKIP)
        
        # 3. EpisodicLifeEnv: Make end-of-life == end-of-episode
        env = EpisodicLifeEnv(env)
        
        # 4. FireResetEnv: Press FIRE button to start games that require it
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        # 5. WarpFrame: Convert to grayscale and resize
        env = WarpFrame(env, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)
        
        # 6. ClipRewardEnv: Clip rewards to {-1, 0, 1}
        env = ClipRewardEnv(env)
        
        # 7. FrameStack: Stack the most recent frames
        env = FrameStack(env, k=config.FRAME_STACK)
        
        # 8. ScaledFloatFrame: Normalize pixel values
        env = ScaledFloatFrame(env)
        
        return env
        
    except gym.error.NamespaceNotFound:
        print("\nERROR: Atari Learning Environment (ALE) not found.")
        print("Please install the required packages with these commands:")
        print("pip install ale-py gymnasium[atari,accept-rom-license]")
        print("OR")
        print("pip install ale-py")
        print("pip install gymnasium[atari]")
        print("python -m ale_py.roms --install-dir <path-to-roms>\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")
        print("Please make sure all dependencies are installed correctly.\n")
        sys.exit(1)


if __name__ == "__main__":
    # Test environment setup
    print("Attempting to create the Ice Hockey environment with wrappers...")
    env = make_env()
    print(f"Environment: {config.ENV_NAME}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Try stepping through the environment
    obs, info = env.reset()
    print(f"Observation shape after preprocessing: {obs.shape}")
    print(f"Observation min/max values: {obs.min():.4f}/{obs.max():.4f}")
    
    for _ in range(10):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {terminated or truncated}")
        if terminated or truncated:
            break
            
    env.close()
    print("Environment test successful!")
