from gym.envs.registration import register

register(
    id='CarRacing-v1',
    entry_point='carracing_gym.envs:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900,
)
