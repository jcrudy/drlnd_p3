from .base import Environment, ClosedEnvironmentError
from unityagents import UnityEnvironment
from abc import abstractclassmethod, abstractproperty, abstractmethod
from . import resources
import os
from .base import IntInterval, CartesianProduct,\
    FloatInterval, MultiEnvironmentMixin
import numpy as np

class UnityBasedEnvironment(Environment):
    '''
    A severe limitation of the UnityEnvironment in unityagents is that only one can be substantiated in 
    a given Python process and it is incompatible with the multiprocessing package, perhaps due to its 
    use of gRPC (see https://github.com/Unity-Technologies/ml-agents/issues/956).
    
    Somehow, the nose module's multiprocess plugin works around these issues, but I don't understand how 
    and can't recreate it.  Therefore, only one UnityBasedEnvironment can be instantiated in the entire 
    history of a particular Python process.
    '''
    @abstractclassmethod
    def path(self):
        '''
        Subclasses should set this to be the path to the 
        desired Unity environment.
        '''
    
    # It's necessary for UnityEnvironments to have unique worker_ids.  We can 
    # ensure no two environments in a process share the same worker id by keeping
    # a count.
    worker_count = 0
    
    def __del__(self):
        UnityBasedEnvironment.worker_count -= 1
    
    def __init__(self, graphics=False):
        # Attempt to make worker_id differ across processes.  Obviously not guaranteed,
        # especially if each process has multiple workers.
        worker_id = (os.getpid() % 100) + UnityBasedEnvironment.worker_count
        self.env = UnityEnvironment(
                                    file_name=self.path, 
                                    no_graphics=(not graphics),
                                    worker_id=worker_id,
#                                     docker_training=True,
                                    )
        UnityBasedEnvironment.worker_count += 1
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        example_state = env_info.vector_observations[0]
        self._state_size = len(example_state)
        self._n_actions = self.brain.vector_action_space_size
        self.closed = False
    
    def reset(self, train):
        if self.closed:
            raise ClosedEnvironmentError('Environment is already closed.')
        env_info = self.env.reset(train_mode=train)[self.brain_name]
        return np.array(env_info.vector_observations)
    
    @abstractmethod
    def transform_action(self, action):
        pass
    
    def step(self, action):
        if self.closed:
            raise ClosedEnvironmentError('Environment is already closed.')
        env_info = self.env.step(self.transform_action(action))[self.brain_name]
        state = np.array(env_info.vector_observations)
        reward = np.array(env_info.rewards)
        done = np.array(env_info.local_done)
        return state, reward, done
    
    def close(self):
        if self.closed:
            return
        self.env.close()
        self.closed = True
    
class BananaEnvironment(UnityBasedEnvironment):
    path = resources.banana
    
    def transform_action(self, action):
        return action
    
    @property
    def action_space(self):
        return CartesianProduct([IntInterval(0, self._n_actions - 1)])
    
    @property
    def state_space(self):
        return CartesianProduct([FloatInterval()] * 37)

class ReacherEnvironmentBase(UnityBasedEnvironment):
    def transform_action(self, action):
        return np.tanh(action)

class ReacherV1Environment(ReacherEnvironmentBase):
    path = resources.reacher_v1
    
    @property
    def action_space(self):
        return CartesianProduct([FloatInterval()] * 4)
    
    @property
    def state_space(self):
        return CartesianProduct([FloatInterval()] * 33)
    
class ReacherV2Environment(ReacherEnvironmentBase, MultiEnvironmentMixin):
    path = resources.reacher_v2
    def __init__(self, graphics=False):
        UnityBasedEnvironment.__init__(self, graphics=graphics)
        
    @property
    def action_space(self):
        return CartesianProduct([[FloatInterval()] * 4] * 20)

    @property
    def state_space(self):
        return CartesianProduct([[FloatInterval()] * 33] * 20)

class TennisEnvironment(UnityBasedEnvironment):
    path = resources.tennis
    
    def transform_action(self, action):
        return action
    
    @property
    def action_space(self):
        return CartesianProduct([IntInterval(0, self._n_actions - 1)])
    
    