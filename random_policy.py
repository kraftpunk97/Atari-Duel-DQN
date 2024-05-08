# Implements a random policy.
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

demo_dict = {
    'Pong': 'PongNoFrameskip-v4',
    'Space Invaders': 'SpaceInvadersNoFrameskip-v4',
    'Breakout': 'BreakoutNoFrameskip-v4'
}
envname = demo_dict['Pong']

env = gym.make(envname, render_mode='human')
env = AtariPreprocessing(env, grayscale_obs=True,
                        scale_obs=True,
                        terminal_on_life_loss=False)
env = FrameStack(env, num_stack=4)
env.metadata['render_fps'] = 30
env.reset()

reward_total = 0
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    reward_total += reward

    if terminated or truncated:
        env.reset()
        print("Reward of this eps", reward_total)
        reward_total = 0


env.close()
