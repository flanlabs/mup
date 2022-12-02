# Copyright 2022 Microsoft Corporation.
from copy import deepcopy

import yaml
from torch import nn
from torch.nn import Linear
from torch.nn.modules.conv import _ConvNd

from mup.infshape import InfShape, zip_infshape
from mup.layer import MuReadout, rescale_linear_bias, INFSHAPE_DICT_KEY, get_infshape_of_param_name
import functools

__BSH_COMMENT__ = '''\
# This is a base shape file encoded in yaml
# - `null` indicates a dimension is "finite", i.e. a non-"width" dimension
# - a number indicates the base dimension of an "infinite" dimension, i.e. some notion of "width"
'''


def get_shapes(model: nn.Module):
    # return {name: param.shape for name, param in model.named_parameters()}

    # Note: this implementation puts duplicate entries for tied weights. That's intended.
    res = {}
    for m_name, m in model.named_modules():
        for p_name, p in m.named_parameters(recurse=False):
            res[m_name+"."+p_name] = p.shape
    
    return res

# def get_shapes(model):
#     shape_dicts = [m for m in model.modules()]
#     all_dicts = functools.reduce(lambda acc, x: acc | x, shape_dicts, {})
#     return all_dicts


#     return {name: param.shape for name, param in model.named_parameters()}

# def _get_last_part(path: str):


def get_infshapes(model: nn.Module):
    res = {}
    for m_name, m in model.named_modules():
        infshapes = getattr(m, INFSHAPE_DICT_KEY)
        for (p_name, shape) in infshapes.items():
            res[m_name+"."+p_name] = shape
    
    return res


    # shape_dicts = [getattr(m, INFSHAPE_DICT_KEY).items() for (name, m) in model.named_modules()]
    # all_dicts = functools.reduce(lambda acc, x: acc | x, shape_dicts, {})
    # print(all_dicts)
    # return all_dicts

    # return {name: param.infshape for name, param in model.named_parameters()}

def save_base_shapes(model_or_shapes, file):
    if isinstance(model_or_shapes, nn.Module):
        sh = get_infshapes(model_or_shapes)
    elif isinstance(model_or_shapes, dict):
        sh = deepcopy(model_or_shapes)
    else:
        raise ValueError()
    sh = {k: s.base_shape() for k, s in sh.items()}
    s = yaml.dump(sh, None, indent=4)
    s = __BSH_COMMENT__ + s
    with open(file, 'w') as f:
        f.write(s)
    
def load_base_shapes(filename):
    '''Get a dict of `InfShape` from a filename.'''
    with open(filename, 'r') as f:
        d = yaml.safe_load(f)
    return {k: InfShape.from_base_shape(v) for k, v in d.items()}

def _dataparallel_hack(base_shapes, shapes):
    '''Fix module name discrepancy caused by (Distributed)DataParallel module.

    The parameters of a (Distributed)DataParallel module all have names that
    start with 'module'. This causes a mismatch from non-DataParallel modules.
    This function tries to match `base_shapes` to `shapes`: if the latter starts
    with 'module', then make the former too; likewise if not.
    '''
    if all(k.startswith('module.') for k in shapes) and \
        all(not k.startswith('module.') for k in base_shapes):
        return {'module.' + k: v for k, v in base_shapes.items()}, shapes
    if all(not k.startswith('module.') for k in shapes) and \
        all(k.startswith('module.') for k in base_shapes):
        return {k.strip('module.'): v for k, v in base_shapes.items()}, shapes
    return base_shapes, shapes


def _extract_shapes(x):
    '''
    Input:
        x: can be any of the following:
            - `nn.Module`
            - dict of shapes
            - dict of `InfShape`
            - str of path to a base shapes (.bsh) file
    Output:
        If `x` is dict of `InfShape`, then output itself.
        If `x` is path, then output a dict of `InfShapes` loaded from `x`.
        Else, output the shapes (not `InfShape`) associated to `x`
    '''
    if isinstance(x, nn.Module):
        x_shapes = get_shapes(x)
    elif isinstance(x, dict):
        x_shapes = deepcopy(x)
    elif isinstance(x, str):
        # x is file name
        x_shapes = load_base_shapes(x)
    else:
        raise ValueError(f'unhandled x type: {type(x)}')
    return x_shapes

def _zip_infshape_dict(base_shapes, shapes):
    '''make a dict of `InfShape` from two dicts of shapes.
    Inputs:
        base_shapes: dict of base shapes or InfShape objects
        shapes: dict of shapes
    Output:
        dict of `InfShape` using `zip_infshape`
    '''
    base_shapes, shapes = _dataparallel_hack(base_shapes, shapes)
    basenames = set(base_shapes.keys())
    names = set(shapes.keys())
    assert basenames == names, (
        f'`base_shapes` has extra names {basenames - names}. '
        f'`shapes` has extra names {names - basenames}.'
    )
    infshapes = {}
    for name, bsh in base_shapes.items():
        infshapes[name] = zip_infshape(bsh, shapes[name])
    return infshapes

def zip_infshapes(base, target):
    '''make a dict of `InfShape` from models or dicts.
    Inputs:
        base: a base `nn.Module` or a dict of shapes
        target: a target `nn.Module` or a dict of shapes
    Output:
        dict of `InfShape` using `zip_infshape`
    '''
    base_shapes = _extract_shapes(base)
    target_shapes = _extract_shapes(target)
    return _zip_infshape_dict(base_shapes, target_shapes)

def clear_dims(infshape_dict):
    '''
    Input:
        infshape_dict: dict of `InfShape`
    Output:
        the same dict but where all `InfDim` in all `InfShape`
        have their `dim` attribute set to None
    '''
    d = deepcopy(infshape_dict)
    for _, v in d.items():
        for infdim in v:
            infdim.dim = None
    return d

def make_base_shapes(base_shapes, delta_shapes, savefile=None):
    '''Make a base shape object from a base model/shapes and a delta model/shapes.

    Inputs:
        base:
            a base `nn.Module` or a dict of shapes
        delta:
            a "delta" model or a dict of shapes, for the sole purpose of
            determining which dimensions are "width" and will be scaled up and
            down in the target model.
        savefile:
            if a string, then the resulting base shape object is serialized to
            this location via yaml encoding.
    Outputs:
        base infshapes
    '''
    bsh = clear_dims(zip_infshapes(base_shapes, delta_shapes))
    if savefile is not None:
        save_base_shapes(bsh, savefile)
    return bsh


def apply_infshapes(model: nn.Module, infshapes):
    for m_name, m in model.named_modules():
        param_shapes = {}
        # print(m)
        for name, p in m.named_parameters(recurse=False):
            full_name = m_name+"."+name
            # print(name)

            # TODO: what if it's a readout? 
            param_shapes[name] = infshapes[full_name]
        setattr(m, INFSHAPE_DICT_KEY, param_shapes)

    # print(get_infshapes(model))

    # for name, p in model.named_parameters():
        
    #     # print(name)
    #     # print(p.)
    #     p.infshape = infshapes[name]

def set_base_shapes(model, base, rescale_params=True, delta=None, savefile=None, do_assert=True):
    '''Sets the `p.infshape` attribute for each parameter `p` of `model`.

    Inputs:
        model: nn.Module instance
        base: The base model.
            Can be nn.Module, a dict of shapes, a str, or None.
            If None, then defaults to `model`
            If str, then treated as filename for yaml encoding of a dict of base shapes.
        rescale_params:
            assuming the model is initialized using the default pytorch init (or
            He initialization etc that scale the same way with fanin): If True
            (default), rescales parameters to have the correct (μP) variances.
        do_assert: 
    Output:
        same object as `model`, after setting the `infshape` attribute of each parameter.
    '''
    if base is None:
        base = model
    base_shapes = _extract_shapes(base)
    if delta is not None:
        delta_shapes = _extract_shapes(delta)
        base_shapes = _zip_infshape_dict(base_shapes, delta_shapes)
    shapes = get_shapes(model)
    infshapes = _zip_infshape_dict(base_shapes, shapes)
    if savefile is not None:
        save_base_shapes(infshapes, savefile)
    apply_infshapes(model, infshapes)
    if do_assert:
        assert_hidden_size_inf(model)
    if rescale_params:
        for name, module in model.named_modules():
            if isinstance(module, MuReadout):
                module._rescale_parameters()
            elif isinstance(module, (Linear, _ConvNd)):
                rescale_linear_bias(module)
    return model


def assert_hidden_size_inf(model):
    '''
    This tests for any `nn.Linear` whose output dimension is finite but input
    dimension is infinite and is not of type `MuReadout`. Such `nn.Linear`
    modules should not exist in a correctly parametrized models.
    '''
    for name, module in model.named_modules():
        if isinstance(module, Linear) and not isinstance(module, MuReadout):
            infshapes = get_infshape_of_param_name(module, "weight")
            if not infshapes[0].isinf() and infshapes[1].isinf():
                assert False, (
                    f'{name} has infinite fan-in and finite fan-out dimensions but is not type `MuReadout`. '
                    'To resolve this, either change the module to `MuReadout` or change the fan-out to an infinite dimension.'
                )
