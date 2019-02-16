from mappo.environment.base import create_tennis_environment
from nose.tools import assert_equal
import numpy as np

def test_tennis_environment():
    env = create_tennis_environment()
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    n_agents = len(env_info.agents)
    assert_equal(n_agents, 2)
    assert_equal(brain.vector_action_space_size, 2)
    states = env_info.vector_observations
    assert_equal(states.shape, (n_agents, 24))
    
    # Do some random episodes.
    for _ in range(5):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(shape=n_agents)
        while True:
            actions = np.tanh(np.random.normal(size=(n_agents, brain.vector_action_space_size)))
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            states = next_states
            if np.all(dones):
                break
            
            
if __name__ == '__main__':
    # Run the tests in this file.
    import sys
    import nose
    module_name = sys.modules[__name__].__file__
 
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s','-v'])