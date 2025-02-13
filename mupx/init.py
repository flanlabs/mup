# Copyright 2022 Microsoft Corporation.
'''
Initializer functions mirroring those of `torch.nn.init`. They serve as
drop-in replacements after the user has called `set_base_shapes` on their
model.

All of the initializers here are designed to 1) behave exactly the same
as the torch versions when the model shapes are equal to their base shapes,
and 2) to scale with width correctly (according to μP), when the model shapes
differ from the base shapes. In general, this means deviating from the
torch version behaviors.
'''
import math
import warnings

import torch
from torch import nn
from torch.nn.init import (_calculate_correct_fan,
                           _calculate_fan_in_and_fan_out, _no_grad_fill_,
                           _no_grad_normal_, _no_grad_uniform_, calculate_gain)

from mupx.layer import get_infshape_of_param

# def assert_infshape_exists(module: nn.Module, tensor: torch.Tensor):



def constant_std_init_(module: nn.Module, tensor: torch.Tensor, sampler_):
    infshape = get_infshape_of_param(module, tensor)
    if infshape.ninf() <= 1:
        sampler_(tensor)
    elif infshape.ninf() == 2:
        sampler_(tensor, scale=infshape.width_mult()**-0.5)
    else:
        raise NotImplementedError()
    return tensor

def uniform_(module: nn.Module, tensor: torch.Tensor, a=0, b=1):
    '''Drop-in replacement of `torch.nn.init.uniform_`.
    Note:
        -  if using this function, ensure `a` and `b` do not depend on fan-in,
           fan-out, or other notions of width, e.g. if a = 0, b = 1.
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    infshape = get_infshape_of_param(module, tensor)
    if a != -b:
        assert infshape.ninf() == 1, 'Sampler for (inf, inf) tensors should have mean 0'
    def sampler_(tensor, scale=1):
        _no_grad_uniform_(tensor, a * scale, b * scale)
    return constant_std_init_(module, tensor, sampler_)

def normal_(module: nn.Module, tensor: torch.Tensor, mean=0, std=1):
    '''Drop-in replacement of `torch.nn.init.normal_`.
    Note:
        -  if using this function, ensure `mean` and `std` do not depend on
           fan-in, fan-out, or other notions of width, e.g. if mean = 0, std =
           1.
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    infshape = get_infshape_of_param(module, tensor)
    if mean != 0:
        assert infshape.ninf() == 1, 'Sampler for (inf, inf) tensors should have mean 0'
    def sampler_(tensor, scale=1):
        _no_grad_normal_(tensor, mean=mean*scale, std=std*scale)
    return constant_std_init_(module, tensor, sampler_)

def ones_(module: nn.Module, tensor: torch.Tensor):
    '''Same as `torch.nn.init.ones_`.
    Note:
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    infshape = get_infshape_of_param(module, tensor)
    assert infshape.ninf() == 1, 'Sampler for (inf, inf) tensors should have mean 0'
    def sampler_(tensor, scale=1):
        _no_grad_fill_(tensor, scale)
    return constant_std_init_(module, tensor, sampler_)

def eye_(module: nn.Module, tensor: torch.Tensor):
    '''Same as `torch.nn.init.eye_`.
    Note:
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    infshape = get_infshape_of_param(module, tensor)
    assert infshape.ninf() == 1, 'Sampler for (inf, inf) tensors should have mean 0'
    return torch.nn.init.eye_(tensor)


def _inf_fan_adjust_xavier(scale, module: nn.Module, tensor: torch.Tensor):
    infshape = get_infshape_of_param(module, tensor)
    fan_out, fan_in = infshape[:2]
    # following are needed to accomodate SP models where all infshapes are finite so base_dims are Nones
    fan_out_base_dim = fan_out.base_dim or fan_out.dim
    fan_in_base_dim = fan_in.base_dim or fan_in.dim
    scale *= math.sqrt(
        (fan_out.dim + fan_in.dim)
        / (fan_out_base_dim + fan_in_base_dim))
    if infshape.ninf() <= 1:
        # should have fixed scale
        pass
    elif infshape.ninf() == 2:
        # should scale like fanin
        assert fan_out.isinf() and fan_in.isinf()
        scale /= math.sqrt(fan_in.width_mult())
    else:
        raise NotImplementedError('can only handle 2 inf dimensions currently')
    return scale


def xavier_uniform_(module: nn.Module, tensor: torch.Tensor, gain=1.):
    '''Drop-in replacement of `torch.nn.init.xavier_uniform_`.
    Note:
        -  if using this function, ensure `gain` does not depend on fan-in,
           fan-out, or other notions of width, e.g. if gain = 1.
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    # _ = get_infshape_of_param(module, tensor)
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    std = _inf_fan_adjust_xavier(std, module, tensor)
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return _no_grad_uniform_(tensor, -a, a)


def xavier_normal_(module: nn.Module, tensor: torch.Tensor, gain=1.):
    '''Drop-in replacement of `torch.nn.init.xavier_normal_`.
    Note:
        -  if using this function, ensure `gain` does not depend on fan-in,
           fan-out, or other notions of width, e.g. if gain = 1.
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    std = _inf_fan_adjust_xavier(std, module, tensor)
    return _no_grad_normal_(tensor, 0., std)


def _inf_fan_adjust_kaiming(scale, module: nn.Module, tensor: torch.Tensor, mode):
    infshape = get_infshape_of_param(module, tensor)
    fan_out, fan_in = infshape[:2]
    if infshape.ninf() == 0:
        return scale
    elif infshape.ninf() == 1:
        # should have fixed scale
        if mode == 'fan_in' and fan_in.isinf():
            scale *= fan_in.width_mult()**0.5
        elif mode == 'fan_out' and fan_out.isinf():
            scale *= fan_out.width_mult()**0.5
    elif infshape.ninf() == 2:
        # should scale like fanin
        assert fan_out.isinf() and fan_in.isinf()
        if mode == 'fan_out':
            scale *= math.sqrt(fan_out.width_mult() / fan_in.width_mult())
    else:
        raise NotImplementedError('can only handle <=2 inf dimensions currently')
    return scale

def kaiming_normal_(module: nn.Module, tensor: torch.Tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    '''Drop-in replacement of `torch.nn.init.kaiming_normal_`.
    Note:
        -  if using this function, ensure `a` does not depend on fan-in,
           fan-out, or other notions of width, e.g. if a = 0.
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = _inf_fan_adjust_kaiming(gain / math.sqrt(fan), module, tensor, mode)
    with torch.no_grad():
        return tensor.normal_(0, std)


def kaiming_uniform_(module: nn.Module, tensor: torch.Tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    '''Drop-in replacement of `torch.nn.init.kaiming_uniform_`.
    Note:
        -  if using this function, ensure `a` does not depend on fan-in,
           fan-out, or other notions of width, e.g. if a = 0.
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = _inf_fan_adjust_kaiming(gain / math.sqrt(fan), module, tensor, mode)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


try:
    from torch.nn.init import _no_grad_trunc_normal_
    def trunc_normal_(module: nn.Module, tensor: torch.Tensor, mean=0, std=1, a=-2, b=2):
        '''Drop-in replacement of `torch.nn.init.trunc_normal_`.
        Note:
            -  if using this function, ensure `mean`, `std`, `a`, `b` do not
               depend on fan-in, fan-out, or other notions of width, e.g. if
               mean = 0, std = 1, a = -2, b = 2.
            - `tensor` should have `infshape` attribute set by
              `set_base_shapes`.
        '''
        infshape = get_infshape_of_param(module, tensor)
        if mean != 0 or a != -b:
            assert infshape.ninf() == 1, 'Sampler for (inf, inf) tensors should have mean 0'
        def sampler_(tensor, scale=1):
            _no_grad_trunc_normal_(tensor, mean=mean*scale, std=std*scale, a=a*scale, b=b*scale)
        return constant_std_init_(module, tensor, sampler_)
except:
    warnings.warn(
        'Failed to import _no_grad_trunc_normal_ from torch.nn.init; '
        'you might be running an older version of torch. trunc_normal_ will not work.')
    def trunc_normal_(tensor, mean=0, std=1, a=-2, b=2):
        warnings.warn('Please upgrade your Pytorch version before using truncated normal.')
        pass
