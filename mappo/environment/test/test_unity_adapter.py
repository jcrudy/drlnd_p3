from mappo.environment.unity_adapter import BananaEnvironment, ReacherV1Environment,\
    ReacherV2Environment, TennisEnvironment
import numpy as np
from nose.tools import assert_equal

def test_banana_environment():
    env = BananaEnvironment()
    state = env.reset(True)
    action = env.action_space.random()#np.random.choice(env.n_actions)
    next_state, reward, done = env.step(action)
    assert_equal(state.shape, next_state.shape)
    assert_equal(reward.dtype, np.float64)
    assert_equal(done.dtype, np.bool)
    env.close()
 
def test_reacher_v1_environment():
    env = ReacherV1Environment()
    state = env.reset(True)
    action = env.action_space.random()#np.random.normal(size=env.n_actions)
    next_state, reward, done = env.step(action)
    assert_equal(state.shape, next_state.shape)
    assert_equal(reward.dtype, np.float64)
    assert_equal(done.dtype, np.bool)
    env.close()

def test_reacher_v2_environment():
    env = ReacherV2Environment()
    state = env.reset(True)
    action = env.action_space.random()#np.random.normal(size=(20, env.n_actions))
    next_state, reward, done = env.step(action)
    assert_equal(state.shape, next_state.shape)
    assert_equal(reward.dtype, np.float64)
    assert_equal(done.dtype, np.bool)
    env.close()

def test_tennis_environment():
    env = TennisEnvironment()

if __name__ == '__main__':
    # Run the tests in this file.
    import sys
    import nose
    module_name = sys.modules[__name__].__file__
 
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s','-v',
                            '--processes=1', 
                            '--process-restartworker'
                            ])
