from abc import abstractmethod, ABCMeta, abstractproperty
from six import with_metaclass, class_types, integer_types
import numpy as np
from itertools import product
from toolz.functoolz import compose
from infinity import inf

class ClosedEnvironmentError(Exception):
    pass

class SpaceValueError(Exception):
    '''
    Raised if an action is not a valid value in the relevant action space.
    '''

class Space(object):
    @abstractmethod
    def validate(self, action):
        '''
        Raise SpaceValueError if action is not valid in this action space.
        '''
    
    @abstractproperty
    def shape(self):
        '''
        The shape of the required actions.
        '''
    
    def __contains__(self, action):
        try:
            self.validate(action)
        except SpaceValueError:
            return False
        return True
    
    @abstractmethod
    def random(self):
        '''
        Return a random action in this action space.
        '''
    
    @abstractproperty
    def size(self):
        '''
        The size (in an appropriate measure) of this space.
        '''
    
    @abstractproperty
    def dtype(self):
        '''
        The dtype of the output.
        '''

class Interval(Space):
    def __init__(self, lower=-inf, upper=inf, lower_closed=True, upper_closed=True):
        self.lower = lower
        self.upper = upper
        self.lower_closed = lower_closed
        self.upper_closed = upper_closed
        # Require element_type to be defined to instantiate
        type(self).element_type  # @UndefinedVariable
    
    @property
    def shape(self):
        return tuple()
    
    def __str__(self):
        left = '[' if self.lower_closed else '('
        right = ']' if self.upper_closed else ')'
        return left + repr(self.lower) + ',' + repr(self.upper) + right
    
    def validate(self, action):
        if not isinstance(action, type(self).element_type):
            raise SpaceValueError('Action {} is not an integer.'.format(action))
        if (action < self.lower or action > self.upper or 
            ((not self.lower_closed) and action == self.lower) or
            ((not self.upper_closed) and action == self.upper)):
            raise SpaceValueError('Action {} is not in the {} {}.'.format(action, type(self).__name__,
                                                                                str(self)))

class IntInterval(Interval):
    element_type = tuple(integer_types) + (np.number,)
    
    def random(self):
        return np.random.randint(self.lower + (1 if not self.lower_closed else 0),
                                 self.upper - (1 if not self.lower_closed else 0))
    
    @property
    def size(self):
        if self.lower > -inf and self.upper < inf:
            lower = self.lower + (1 if not self.lower_closed else 0)
            upper = self.upper - (1 if not self.lower_closed else 0)
            return upper - lower + 1
        else:
            return inf
    
    @property
    def dtype(self):
        return np.int64
        
class FloatInterval(Interval):
    element_type = (float, np.floating)

    def random(self):
        if self.lower > -inf and self.upper < inf:
            return np.random.uniform(self.lower + (1 if not self.lower_closed else 0),
                                     self.upper - (1 if not self.lower_closed else 0))
        elif self.lower > -inf:
            return self.lower + np.random.exponential()
        elif self.upper < inf:
            return self.upper - np.random.exponential()
        else:
            return np.random.normal()
    
    @property
    def size(self):
        if self.lower > -inf and self.upper < inf:
            return self.upper - self.lower
        else:
            return inf
    
    @property
    def dtype(self):
        return np.float64

class CartesianProduct(Space):
    def __init__(self, set_array):
        self.set_array = np.asarray(set_array)
    
    @property
    def shape(self):
        return self.set_array.shape
    
    def validate(self, action):
        action = np.asarray(action)
        if action.shape != self.shape:
            raise SpaceValueError('Action shape {} does not match required shape {}.'.format(action.shape, self.set_array.shape))
        
        for coord in product(*map(compose(tuple, range), self.shape)):
            print(coord)
            action_element = action[coord]
            set_element = self.set_array[coord]
            set_element.validate(action_element)
    
    def random(self):
        result = np.empty_like(self.set_array)
        dtypes = set()
        for coord in product(*map(compose(tuple, range), self.shape)):
            result[coord] = self.set_array[coord].random()
            dtypes.add(self.set_array[coord].dtype)
        if len(dtypes) == 1:
            result = result.astype(next(iter(dtypes)))
        return result
    
    @property
    def size(self):
        result = 1
        for coord in product(*map(compose(tuple, range), self.shape)):
            result *= self.set_array[coord].size
        return result
    
class Environment(with_metaclass(ABCMeta, object)):
    @abstractproperty
    def action_space(self):
        '''
        The action Space for the environment.  For performance, it's probably best for agents to use the 
        action_space during initialization, but not to validate every action.  For debugging, action 
        validation is a good idea.
        '''
    
    @abstractproperty
    def state_space(self):
        '''
        The state Space for the environment.
        ''' 
    
    @abstractmethod
    def reset(self, train):
        '''
        Start a new episode and return the initial state.
        '''
    
    @abstractmethod
    def step(self, action):
        '''
        Take action and return (state, reward, done) tuple.
        '''
    
    @abstractmethod
    def close(self):
        '''
        Close the environment.
        '''

class MultiEnvironmentMixin(object):
    '''
    An environment which is multiple copies of the same environment.
    '''
# 
# class CompositeEnvironment(MultiEnvironmentMixin, Environment):
#     def __init__(self, environment_generator, n_copies):
#         self.n_copies = n_copies
#         self.environment_generator = environment_generator
#         self.environments = tuple(self.environment_generator() for _ in range(self.n_copies))
#     
#     # TODO: Finish this class
    
    