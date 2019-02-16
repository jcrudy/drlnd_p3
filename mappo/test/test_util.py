import numpy as np
from numpy.ma.testutils import assert_array_almost_equal
from a2c.util import discount, td_target, gae

def test_discount():
    trials = 10
    time_steps = 100
    
    rewards = np.random.normal(size=(trials, time_steps))
    gamma = .9
    
    expected_result = np.empty(shape=(trials, time_steps))
    
    for i in range(time_steps):
        discount_vec = gamma ** np.arange(time_steps - i)
        expected_result[:, i] = np.dot(rewards[:, i:], discount_vec)
    
    assert_array_almost_equal(expected_result, discount(gamma, rewards))
        
def test_discount_with_extra_dim():
    trials = 10
    time_steps = 100
    
    rewards = np.random.normal(size=(trials, time_steps, 1))
    gamma = .9
    
    expected_result = np.empty(shape=(trials, time_steps, 1))
    
    for i in range(time_steps):
        discount_vec = gamma ** np.arange(time_steps - i)
        expected_result[:, i, 0] = np.dot(rewards[:, i:, 0], discount_vec)
    
    assert_array_almost_equal(expected_result, discount(gamma, rewards))

def test_td_target():
    trials = 10
    time_steps = 100
    
    rewards = np.random.normal(size=(trials, time_steps, 1))
    values = np.random.normal(size=(trials, time_steps, 1))
    gamma = .9
    
    expected_result = rewards.copy()
    expected_result[:, :-1, :] += gamma * values[:, 1:, :]
    
    assert_array_almost_equal(expected_result, td_target(gamma, rewards, values))

def test_gae():
    trials = 2
    time_steps = 2
    
    rewards = np.random.normal(size=(trials, time_steps, 1))
    values = np.random.normal(size=(trials, time_steps, 1))
    gamma = .9
    lambda_ = .5
    
    advantage = np.zeros_like(rewards)
    for t1 in range(time_steps):
        for t2 in range(t1, time_steps):
            l = t2 - t1
            delta = rewards[:, t2, 0] + (gamma * values[:, t2 + 1, 0] if t2 < (time_steps - 1) else 0.) - values[:, t2, 0]
            advantage[:, t1, 0] += ((gamma * lambda_) ** l) * delta
            
    assert_array_almost_equal(advantage, gae(gamma, lambda_, rewards, values))
    

if __name__ == '__main__':
    # Run the tests in this file.
    import sys
    import nose
    module_name = sys.modules[__name__].__file__
 
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s','-v'])