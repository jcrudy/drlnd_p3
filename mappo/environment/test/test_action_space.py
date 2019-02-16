from deeprl.environment.base import CartesianProduct,\
    SpaceValueError, FloatInterval, IntInterval
from nose.tools import assert_raises
import numpy as np

def test_cartesian_product_action_space():
    action_space = CartesianProduct(
                       [[IntInterval(0, 10), FloatInterval(0., 10.)], 
                        [FloatInterval(-10., 0.), IntInterval(-10, 0)]])
    valid_action = np.array([[2, 2.], [-2.5, -2]], dtype=object)
    invalid_action = np.array([[2.5, 2], [-2, -2.5]], dtype=object)
    assert(valid_action in action_space)
    assert(invalid_action not in action_space)
    assert_raises(SpaceValueError, action_space.validate, invalid_action)
    
    assert_raises(AssertionError, assert_raises, SpaceValueError, action_space.validate, valid_action)
    
def test_float_interval():
    space = FloatInterval(0, 1, upper_closed=False)
    assert(np.float64(.5) in space)
    assert(np.float32(.5) in space)
    assert(0.5 in space)
    assert(0 not in space)
    assert(0. in space)
    assert(1. not in space)
    
if __name__ == '__main__':
    # Run the tests in this file.
    import sys
    import nose
    module_name = sys.modules[__name__].__file__
 
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s','-v'])