from gym.envs.registration import register

register(
    id='AdversarialGrid-v0',
    entry_point='gym-adversarialgrid.envs:FooEnv',
)
