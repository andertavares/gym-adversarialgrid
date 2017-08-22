from gym.envs.registration import register

register(
    id='AdversarialGrid-v0',
    entry_point='gym_adversarialgrid.envs:AdversarialGrid',
)

register(
    id='ARATGrid-v0',
    entry_point='gym_adversarialgrid.envs:ARATGrid',
)
