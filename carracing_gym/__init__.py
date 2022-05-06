from gym.envs.registration import register

register(
    id='CarRacing-v1',
    entry_point='carracing_gym.envs:CarRacing',
)
