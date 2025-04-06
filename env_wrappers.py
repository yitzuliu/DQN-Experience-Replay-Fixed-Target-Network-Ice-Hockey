import gymnasium as gym
import config
import sys
import ale_py

def make_env(env_name=config.ENV_NAME, render_mode=config.RENDER_MODE):
    """
    Create an Atari Ice Hockey environment with no wrappers.
    
    This function creates a raw gymnasium environment for Ice Hockey without 
    any preprocessing or wrappers. The environment will use the original frame
    rate, image dimensions, and reward structure of the Atari game.
    
    Args:
        env_name (str): Name of the Atari environment to create
            (default: value from config.ENV_NAME)
        render_mode (str): Rendering mode, None for no rendering or 'human' for
            visualization (default: value from config.RENDER_MODE)
        
    Returns:
        gym.Env: A raw gymnasium environment for DQN training
    """
    try:
        # Try to create the environment
        env = gym.make(env_name, render_mode=render_mode)
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
    print("Attempting to create the Ice Hockey environment...")
    env = make_env()
    print(f"Environment: {config.ENV_NAME}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Try stepping through the environment
    obs, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {terminated or truncated}")
        if terminated or truncated:
            break
            
    env.close()
    print("Environment test successful!")
