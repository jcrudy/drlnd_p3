import numpy as np
from itertools import repeat
import torch
from torch import nn
from copy import deepcopy
from six import integer_types
from multipledispatch.dispatcher import Dispatcher
from torch.tensor import Tensor
from numpy import ndarray
import scipy.signal
import os
from typing import Iterable


class Constant(object):
    def __init__(self, value):
        self.value = value
        
    def __call__(self, *args, **kwargs):
        return self.value

def rolling_mean(y, window):
    result = (
              np.convolve(y, np.ones(shape=window), mode='full') / 
              np.convolve(np.ones_like(y), np.ones(shape=window), mode='full')
              )
    return result[:(1-window)]

class ModuleStack(nn.Module):
    def __init__(self, modules, input_dim=0, output_dim=0):
        super().__init__()
        if isinstance(modules, nn.Module):
            modules = [modules]
        self.parts = nn.ModuleList(modules)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, data):
        results = []
        if len(data.shape) < self.input_dim:
            raise ValueError()
        if data.shape[self.input_dim] != len(self.parts):
            raise ValueError()
        for i, module in enumerate(self.parts):
            slice_tup = (tuple(repeat(slice(None, None, None), self.input_dim)) +
                         (i,) + 
                         tuple(repeat(slice(None, None, None), len(data.shape) - self.input_dim - 1)))
            results.append(module(data[slice_tup]))
        return torch.stack(results, dim=self.output_dim)
    
    @classmethod
    def repeat(cls, module, n_times=1, dim=0):
        return cls(map(deepcopy, repeat(module, n_times)),
                   input_dim=dim, output_dim=dim)
    
    def __iter__(self):
        return iter(self.parts)
    
    def __mul__(self, n):
        if not isinstance(n, integer_types):
            return NotImplemented
        return self.__class__(sum(map(deepcopy, repeat(self.parts, n))),
                   input_dim=self.input_dim, output_dim=self.output_dim)
    
    def __rmul__(self, n):
        if not isinstance(n, integer_types):
            return NotImplemented
        return self.__mul__(n)
    
    def __len__(self):
        return len(self.parts)
    
    def __add__(self, other):
        if isinstance(other, type(self)):
            if self.input_dim != other.input_dim or self.output_dim != other.output_dim:
                raise ValueError('Incompatible input or output dimensions.')
            return self.__class__(deepcopy(tuple(self.parts)) + deepcopy(tuple(other.parts)),
                                  input_dim=self.input_dim, output_dim=self.output_dim)
        else:
            return NotImplemented
    
    def __radd__(self, other):
        return other.__add__(self)
        
torchify = Dispatcher('torchify')

@torchify.register(Tensor)
def torchify_tensor(tens):
    return tens

@torchify.register(ndarray)
def torchify_numpy(arr):
    return torch.from_numpy(arr)

torchify32 = Dispatcher('torchify32')

@torchify32.register(ndarray)
def torchify32_numpy(x):
    # TODO: This could obviously be more efficient.  It 
    # also might be good for it to handle integers differently.
    return torchify(x.astype(np.float32))

@torchify32.register(float)
def torchify32_float(x):
    return torchify32(np.array([x]))[0]

numpify = Dispatcher('numpify')

@numpify.register(float)
def numpify_float(flt):
    return np.float(flt)

@numpify.register(Tensor)
def numpify_tensor(tens):
    return tens.detach().numpy()

@numpify.register(ndarray)
def numpify_numpy(arr):
    return arr
    
def discount(gamma, rewards, axis=1):
    """
    See https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation.
    """
    r = np.flip(rewards, axis=axis)
    a = [1, -gamma]
    b = [1]
    y = scipy.signal.lfilter(b, a, x=r, axis=axis)
    return np.flip(y, axis=axis)

def td_target(gamma, rewards, values, axis=1):
    values_slicer_pre = (slice(None, None, None),) * axis
    values_slicer_post = (slice(None, None, None),) * (len(rewards.shape) - axis - 1)
    values_slicer = values_slicer_pre + (slice(1, None, None),) + values_slicer_post
    pad_shape = values.shape[:axis] + (1,) + values.shape[(axis + 1):]
    shifted_values = np.concatenate([values[values_slicer], np.zeros(shape=pad_shape)], axis=axis)
    return rewards + gamma * shifted_values

def gae_from_td_target(gamma, lambda_, target, values, axis=1):
    return discount(gamma * lambda_, target - values, axis=axis)

def gae(gamma, lambda_, rewards, values, axis=1):
    target = td_target(gamma, rewards, values, axis=axis)# - values
    return gae_from_td_target(gamma, lambda_, target, values, axis=axis)

def split_path(path):
    path, tail = os.path.split(path)
    if not path:
        if not tail:
            return []
        else:
            return [tail]
    return split_path(path) + ([tail] if tail else [])
        

tupify = Dispatcher('tupify')

@tupify.register((np.ndarray, object))
def tupify_ndarray(arr):
    return (arr,)

@tupify.register(Iterable)
def tupify_iterable(itr):
    return tuple(itr)




