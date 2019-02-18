from torch import optim, nn
from functools import partial
from mappo.util import Constant, torchify32, numpify, gae, td_target,\
    rolling_mean, tupify
import numpy as np
from abc import abstractmethod, ABCMeta
import torch
from six import with_metaclass
from torch.distributions.normal import Normal
from torch.nn import functional as F
from multipledispatch.dispatcher import Dispatcher
from typing import Iterable
from toolz.functoolz import curry
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle

class PolicyModel(with_metaclass(ABCMeta, nn.Module)):
    @abstractmethod
    def rv(self, state):
        pass
    
    def sample(self, state, *args, **kwargs):
        return self.rv(state).sample(*args, **kwargs)
    
    def log_prob(self, state, sample):
        return torch.sum(self.rv(state).log_prob(sample), dim=-1)
    
    def prob(self, state, sample):
        return torch.exp(self.log_prob(state, sample))

class MuSigmaLayer(nn.Module):
    def __init__(self, input_size, output_size):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)
    
    def forward(self, input_data):
        mu = self.mu_layer(input_data)
        sigma = F.softplus(self.sigma_layer(input_data))
        return torch.stack([mu, sigma], dim=-1)
    
class NormalPolicy(PolicyModel):
    def __init__(self, network):
        '''
        network (nn.Module): Must end with a MuSigmaLayer or similar.
        '''
        nn.Module.__init__(self)
        self.network = network
    
    def rv(self, state):
        mu_sigma = self.network(state)
        slicer = (slice(None, None, None),) * (len(mu_sigma.shape) - 1)
        mu = mu_sigma[slicer + (0,)]
        sigma = mu_sigma[slicer + (1,)]
        return Normal(mu, sigma)

# class TanhNormalPolicy(NormalPolicy):
#     def sample(self, *args, **kwargs):
#         return F.tanh(NormalPolicy.sample(self, *args, **kwargs))
#     
#     def log_prob(self, state, sample):
#         return NormalPolicy.log_prob(self, state, 0.5 * torch.log((1 + sample) / (1 - sample)))
    
# @curry
# def select_first_dims(selection, arr):
#     return arr[tupify(selection) + (slice(None, None, None),) * (len(arr.shape) - len(selection))]

@curry
def select_dims(selection, left_pad, arr):
    '''
    selection (tuple of pairs): Has form ((i, indices),...).
    '''
    return arr[(slice(None, None, None),) * left_pad + tupify(selection) + 
               (slice(None, None, None),) * (len(arr.shape) - left_pad - len(selection))]
#     selector = [slice(None, None, None)] * len(arr.shape)
#     for i, indices in selection:
#         selector[i] = indices
#     return arr[tuple(selector)]
    
@curry
def select_batch(expected_batch_size, *args):
    '''
    Randomly select episodes and time steps.
    '''
    p = min(expected_batch_size / float(np.prod(args[0].shape[1])), 1.)
    selection = np.where(np.random.binomial(1, p, size=args[0].shape[1]) > 0)
    return tuple(map(select_dims(selection, 1), args))

@curry
def select_agent(agent_index, *args):
    result = []
    for arg in args:
        selector = (agent_index,) + (slice(None,None,None),) * (len(arg.shape) - 1)
        result.append(arg[selector])
    return tuple(result) if len(result) > 1 else result[0]

class Agent(object):
    def __init__(self, policy_model, policy_optimizerer=partial(optim.Adam, lr=2e-4),
                 policy_schedulerer=partial(optim.lr_scheduler.LambdaLR, lr_lambda=Constant(1.))):
        self.policy_model = policy_model
        self.policy_optimizer = policy_optimizerer(self.policy_model.parameters())
        self.policy_scheduler = policy_schedulerer(self.policy_optimizer)
        
    def act(self, state):
        return self.policy_model.sample(state)
    
    def prob(self, state, action):
        return self.policy_model.prob(state, action)
    
class Trainer(object):
    def __init__(self, agents, value_model, value_optimizerer=partial(optim.Adam, lr=3e-4), 
                 value_schedulerer=partial(optim.lr_scheduler.LambdaLR, lr_lambda=Constant(1.)),
                 gamma=.9, lambda_=0., n_updates_per_episode=40, n_episodes_per_batch=100, epsilon=.1, 
                 expected_minibatch_size=750, policy_clip=None, value_clip=None, action_transformer=np.tanh):
        self.agents = agents
        self.value_model = value_model
        self.value_optimizer = value_optimizerer(self.value_model.parameters())
        self.value_scheduler = value_schedulerer(self.value_optimizer)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.n_episodes_per_batch = n_episodes_per_batch
        self.n_updates_per_episode = n_updates_per_episode
        self.train_episodes = []
        self.train_scores = []
        self.epochs_trained = 0
        self.episodes_trained = 0
        self.epsilon = epsilon
        self.expected_minibatch_size = expected_minibatch_size
        self.policy_clip = policy_clip
        self.value_clip = value_clip
        self.action_transformer = action_transformer
    
    
    def to_pickle(self, filename):
        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)
    
    @classmethod
    def from_pickle(cls, filename):
        with open(filename, 'rb') as outfile:
            result = pickle.load(outfile)
        if type(result) is not cls:
            raise TypeError('Unpickled object is not correct type.')
        return result
    
    def value(self, state, action):
        if len(state.shape) == 2:
            return self.value_model(torch.cat([state.view(state.numel()), action.view(action.numel())], dim=0))
        elif len(state.shape) == 3:
            return self.value_model(torch.cat([state.permute([1,0,2]).view(state.shape[1], -1), action.permute([1,0,2]).view(action.shape[1], -1)], dim=-1)).permute([1,0])
        else:
            raise ValueError('Unexpected shape for state: %s' % str(state.shape))
        
    def collect_trajectory(self, environment):
        brain_name = environment.brain_names[0]
        env_info = environment.reset(train_mode=True)[brain_name]
        # The state has shape (agents, observation_size)
        next_state = env_info.vector_observations
        done = [False]
        trajectory = []
        while not np.all(done):
            state = next_state
            torch_state = torchify32(state)
            actions = []
            probs = []
            for i, agent in enumerate(self.agents):
                agent_action = agent.act(torch_state[i,:])
                agent_prob = agent.prob(torch_state[i,:], agent_action)
                actions.append(agent_action)
                probs.append(agent_prob)
            action = torch.stack(actions, dim=0)
            prob = torch.stack(probs, dim=0)
#             action = self.policy_model.sample(torch_state)
#             prob = self.policy_model.prob(torch_state, action)
            value = self.value(torch_state, action)
#             value = self.value_model(torch_state)
            numpy_action = numpify(action)
            env_info = environment.step(self.action_transformer(numpy_action))[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
#             next_state, reward, done = environment.step(numpy_action)
            
            trajectory.append((state, numpy_action, numpify(prob), reward, done, numpify(value)))
        return list(map(partial(np.stack, axis=1), zip(*trajectory)))
    
    def augment_trajectory(self, trajectory):
        states, actions, old_probs, rewards, dones, old_values = trajectory
        td_target_values = td_target(self.gamma, rewards, old_values, axis=1)
        advantages = gae(self.gamma, self.lambda_, rewards, old_values, axis=1)
        advantage_means = np.mean(advantages, axis=1, keepdims=True)
        advantage_sds = np.std(advantages, axis=1, keepdims=True)
        normalized_advantages = (advantages - advantage_means) / np.where(advantage_sds > 1e-6, advantage_sds, 1.)
        return states, actions, old_probs, rewards, dones, old_values, td_target_values, normalized_advantages
    
    def collect_batch(self, environment):
        trajectories = [self.augment_trajectory(self.collect_trajectory(environment)) for _ in range(self.n_episodes_per_batch)]

        # Stack them along the time dimension (1) and return.
        return list(map(partial(np.concatenate, axis=1), zip(*trajectories)))
    
    def train(self, environment, num_epochs=1000):
        for _ in tqdm(range(num_epochs)):
            self.train_step(environment)
            for agent in self.agents:
                agent.policy_scheduler.step(self.train_scores[-1])
            self.value_scheduler.step(self.train_scores[-1])
    
    
    def train_step(self, environment):
        '''
        Assume environment output has shape (n_agents, n_time_steps * episodes, ...)
        '''
        
        # Collect a batch of episodes for all agents.
        states, actions, old_probs, rewards, dones, old_values, td_target_values, normalized_advantages = self.collect_batch(environment)
        
        # Some stats
        total_rewards = np.sum(rewards, axis=1)
        average_rewards = total_rewards / float(self.n_episodes_per_batch)
        score = np.max(average_rewards)
        
#         # Compute TD target and gae.
#         td_target_values = td_target(self.gamma, rewards, old_values, axis=2)
#         advantages = gae(self.gamma, self.lambda_, rewards, old_values, axis=2)
#         advantage_means = np.mean(advantages, axis=2, keepdims=True)
#         advantage_sds = np.std(advantages, axis=2, keepdims=True)
#         normalized_advantages = (advantages - advantage_means) / np.where(advantage_sds > 1e-6, advantage_sds, 1.)
        
        
        for _ in range(self.n_updates_per_episode):
            # Sample minibatch.
            (batch_states, batch_actions, batch_old_probs, 
             batch_td_targets, batch_advantages) = \
                select_batch(self.expected_minibatch_size, states, actions, old_probs, 
                                  td_target_values, normalized_advantages)
            
            batch_torch_states = torchify32(batch_states)
            batch_torch_td_targets = torchify32(batch_td_targets)
            batch_torch_actions = torchify32(batch_actions)
            
            # Update value model.
            value_loss = torch.mean((self.value(batch_torch_states, batch_torch_actions) - batch_torch_td_targets) ** 2)
            self.value_optimizer.zero_grad()
            value_loss.backward()
#             print('value_loss = %s' % value_loss)
            if self.value_clip is not None:
                nn.utils.clip_grad_norm(self.policy_model.parameters(), self.value_clip)
            self.value_optimizer.step()
            
            # Update the policy model for each agent.
            for i, agent in enumerate(self.agents):
                batch_agent_torch_states, batch_agent_torch_actions, batch_agent_old_probs, \
                        batch_agent_advantages = select_agent(i, 
                        batch_torch_states, batch_torch_actions, batch_old_probs, batch_advantages)
                
                batch_agent_probs = agent.prob(batch_agent_torch_states, batch_agent_torch_actions)
                batch_agent_ratio = batch_agent_probs / torchify32(batch_agent_old_probs)
                batch_agent_ratio[batch_agent_ratio != batch_agent_ratio] = 0.
                batch_agent_clipped_ratio = batch_agent_ratio.clamp(min = 1. - self.epsilon, max = 1 + self.epsilon)
                batch_agent_torch_advantages = torchify32(batch_agent_advantages)
                agent_policy_loss = -torch.mean(torch.min(batch_agent_ratio * batch_agent_torch_advantages, 
                                          batch_agent_clipped_ratio * batch_agent_torch_advantages))
#                 print('agent_policy_loss = %s' % agent_policy_loss)
                agent.policy_optimizer.zero_grad()
                agent_policy_loss.backward()
                if self.policy_clip is not None:
                    nn.utils.clip_grad_norm(agent.policy_model.parameters(), self.policy_clip)
                agent.policy_optimizer.step()
                
        
        self.episodes_trained += self.n_episodes_per_batch
        self.epochs_trained += 1
        self.train_episodes.append(self.episodes_trained)
        self.train_scores.append(score)

    def plot(self, epochs=None, moving_window=100):
        x = np.array(self.train_episodes)
        y = np.array(self.train_scores)
        y_mean = rolling_mean(y, moving_window)
        if epochs is not None:
            x = x[-epochs:]
            y = y[-epochs:]
            y_mean = y_mean[-epochs:]
        plt.plot(x, y)
        plt.plot(x, y_mean)
    
# 
# class Agent(object):
#     def __init__(self, policy_model, value_model, policy_optimizerer=partial(optim.Adam, lr=3e-4), 
#                  value_optimizerer=partial(optim.Adam, lr=3e-4),
#                  value_schedulerer=partial(optim.lr_scheduler.LambdaLR, lr_lambda=Constant(1.)),
#                  policy_schedulerer=partial(optim.lr_scheduler.LambdaLR, lr_lambda=Constant(1.)),
#                  gamma=.9, lambda_=0., n_updates_per_episode=10, epsilon=.1, 
#                  expected_minibatch_size=15000, policy_clip=None, value_clip=None):
#         self.policy_model = policy_model
#         self.value_model = value_model
#         self.policy_optimizer = policy_optimizerer(self.policy_model.parameters())
#         self.policy_scheduler = policy_schedulerer(self.policy_optimizer)
#         self.value_optimizer = value_optimizerer(self.value_model.parameters())
#         self.value_scheduler = value_schedulerer(self.value_optimizer)
#         self.gamma = gamma
#         self.lambda_ = lambda_
#         self.n_updates_per_episode = n_updates_per_episode
#         self.train_episodes = []
#         self.train_scores = []
#         self.epochs_trained = 0
#         self.episodes_trained = 0
#         self.epsilon = epsilon
#         self.expected_minibatch_size = expected_minibatch_size
#         self.policy_clip = policy_clip
#         self.value_clip = value_clip
#     
#     def to_pickle(self, filename):
#         with open(filename, 'wb') as outfile:
#             pickle.dump(self, outfile)
#     
#     @classmethod
#     def from_pickle(cls, filename):
#         with open(filename, 'rb') as outfile:
#             result = pickle.load(outfile)
#         if type(result) is not cls:
#             raise TypeError('Unpickled object is not correct type.')
#         return result
#     
#     def collect_trajectory(self, environment):
#         brain_name = environment.brain_names[0]
#         env_info = environment.reset(train_mode=True)[brain_name]
#         next_state = env_info.vector_observations
#         done = [False]
#         trajectory = []
#         while not np.all(done):
#             state = next_state
#             torch_state = torchify32(state)
#             action = self.policy_model.(torch_state)
#             prob = self.policy_model.prob(torch_state, action)
#             value = self.value_model(torch_state).squeeze(-1)
# #             value = self.value_model(torch_state)
#             numpy_action = numpify(action)
#             env_info = environment.step(numpy_action)[brain_name]
#             next_state, reward, done = environment.step(numpy_action)
#             
#             trajectory.append((state, numpy_action, numpify(prob), reward, done, numpify(value)))
#         return list(map(partial(np.stack, axis=1), zip(*trajectory)))
#     
#     def plot(self, epochs=None, moving_window=100):
#         x = np.array(self.train_episodes)
#         y = np.array(self.train_scores)
#         y_mean = rolling_mean(y, moving_window)
#         if epochs is not None:
#             x = x[-epochs:]
#             y = y[-epochs:]
#             y_mean = y_mean[-epochs:]
#         plt.plot(x, y)
#         plt.plot(x, y_mean)
#         
#     
#     def train(self, environment, num_epochs=1000):
#         for _ in tqdm(range(num_epochs)):
#             self.train_step(environment)
#             self.policy_scheduler.step(self.train_scores[-1])
#             self.value_scheduler.step(self.train_scores[-1])
#     
#     def train_step(self, environment):
#         '''
#         Assume environment output has shape (n_agents, n_time_steps, ...)
#         '''
#         # Collect an episode of data for all agents.
#         states, actions, old_probs, rewards, dones, old_values = self.collect_trajectory(environment)
#         
#         # Collect stats.
#         n_episodes = states.shape[0]
#         total_rewards = np.sum(rewards)
#         average_rewards = total_rewards / float(n_episodes)
#         
#         # Compute TD target and gae.
#         td_target_values = td_target(self.gamma, rewards, old_values)
#         advantages = gae(self.gamma, self.lambda_, rewards, old_values)
#         advantage_means = np.mean(advantages, axis=1, keepdims=True)
#         advantage_sds = np.std(advantages, axis=1, keepdims=True)
#         normalized_advantages = (advantages - advantage_means) / np.where(advantage_sds > 1e-6, advantage_sds, 1.)
#         
#         for _ in range(self.n_updates_per_episode):
#             # Sample minibatch.
#             (batch_states, batch_actions, batch_old_probs, 
#              batch_td_targets, batch_advantages) = \
#                 select_batch(self.expected_minibatch_size, states, actions, old_probs, 
#                                   td_target_values, normalized_advantages)
#             
#             batch_torch_states = torchify32(batch_states)
#             batch_torch_td_targets = torchify32(batch_td_targets)
#             batch_torch_actions = torchify32(batch_actions)
#             
#             # Update value model.
#             value_loss = torch.mean((self.value_model(batch_torch_states).squeeze(-1) - batch_torch_td_targets) ** 2)
#             self.value_optimizer.zero_grad()
#             value_loss.backward()
#             if self.value_clip is not None:
#                 nn.utils.clip_grad_norm(self.policy_model.parameters(), self.value_clip)
#             self.value_optimizer.step()
#             
#             # Update the policy model.
#             self.policy_optimizer.zero_grad()
#             
#             batch_probs = self.policy_model.prob(batch_torch_states, batch_torch_actions)
#             batch_ratio = batch_probs / torchify32(batch_old_probs)
#             batch_clipped_ratio = batch_ratio.clamp(min = 1. - self.epsilon, max = 1 + self.epsilon)
#             batch_torch_advantages = torchify32(batch_advantages)
#             policy_loss = -torch.mean(torch.min(batch_ratio * batch_torch_advantages, 
#                                           batch_clipped_ratio * batch_torch_advantages))
#             policy_loss.backward()
#             if self.policy_clip is not None:
#                 nn.utils.clip_grad_norm(self.policy_model.parameters(), self.policy_clip)
#             self.policy_optimizer.step()
#             batch_probs = self.policy_model.prob(batch_torch_states, batch_torch_actions)
#             batch_ratio = batch_probs / torchify32(batch_old_probs)
#             batch_clipped_ratio = batch_ratio.clamp(min = 1. - self.epsilon, max = 1 + self.epsilon)
#             batch_torch_advantages = torchify32(batch_advantages)
#             policy_loss = -torch.mean(torch.min(batch_ratio * batch_torch_advantages, 
#                                           batch_clipped_ratio * batch_torch_advantages))
#         
#         self.episodes_trained += n_episodes
#         self.epochs_trained += 1
#         self.train_scores.append(average_rewards)
#         self.train_episodes.append(self.episodes_trained)
#         
#     
    
    