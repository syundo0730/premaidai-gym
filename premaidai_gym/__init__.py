from gym.envs.registration import register

register(
    id='RoboschoolPremaidAIWalker-v0',
    entry_point='premaidai_gym:RoboschoolPremaidAIWalker',
    max_episode_steps=9999,
    reward_threshold=3500.0,
    tags={"pg_complexity": 100*1000000},
    )

register(
    id='RoboschoolPremaidAIMimicWalker-v0',
    entry_point='premaidai_gym:RoboschoolPremaidAIMimicWalker',
    max_episode_steps=9999,
    reward_threshold=3500.0,
    tags={"pg_complexity": 100*1000000},
)

register(
    id='RoboschoolPremaidAIStabilizationWalker-v0',
    entry_point='premaidai_gym:RoboschoolPremaidAIStabilizationWalker',
    max_episode_steps=9999,
    reward_threshold=3500.0,
    tags={"pg_complexity": 100*1000000},
)

from premaidai_gym.roboschool_premaidai_walker import RoboschoolPremaidAIWalker
from premaidai_gym.roboschool_premaidai_walker import RoboschoolPremaidAIMimicWalker
from premaidai_gym.roboschool_premaidai_walker import RoboschoolPremaidAIStabilizationWalker
