# Copyright 2022 Microsoft Corporation.
'''
Optimizers with μP scaling.

Here we provide 3 ready-to-go optimizers MuAdam, MuAdamW, and MuSGD.
However, the user can easily convert their own optimizer to a μP
optimizer: if your `optimizer` is "Adam-like", such as RMSProp and Adagrad,
that involves normalizing the gradient entrywise, then the following creates
the desired μP optimizer:

    def MuOptimizer(params, **kwargs):
        return MuAdam(params, impl=optimizer, **kwargs)

On the other hand, if your `optimizer` is "SGD-like", such as ASGD, then
the following creates the desired μP optimizer:

    def MuOptimizer(params, **kwargs):
        return MuSGD(params, impl=optimizer, **kwargs)

See Appendix B in our paper for discussions of other optimizers.
'''
from collections import defaultdict
from typing import Dict, List, Sequence

from torch.optim import SGD, Adam, AdamW

from torch import nn
import torch
import torch.distributed.fsdp.flatten_params_wrapper
import composer.optim
import mup.shape

# def process_param_groups(params, model: nn.Module, **kwargs):
#     param_groups = list(params)
#     if not isinstance(param_groups[0], dict):
#         param_groups = [{'params': param_groups}]
#     for param_group in param_groups:
#         if 'lr' not in param_group:
#             param_group['lr'] = kwargs['lr']
#         if 'weight_decay' not in param_group:
#             param_group['weight_decay'] = kwargs.get('weight_decay', 0.)
#     return param_groups



# def expand_flat_params(params: Sequence[torch.Tensor], model: nn.Module):
#     for p in params:

# class MuAdam(Adam):
#     def __init__(self, model: nn.Module, *args, **kwargs):
#         params = []
#         for name, m in model.module

# def MuAdam(params, impl=Adam, decoupled_wd=False, **kwargs):
#     '''Adam with μP scaling.

#     Note for this to work properly, your model needs to have its base shapes set
#     already using `mup.set_base_shapes`.
    
#     Inputs:
#         impl: the specific Adam-like optimizer implementation from torch.optim or
#             elsewhere 
#         decoupled_wd: if True, skips the mup scaling for weight decay, which should
#             be used for optimizer implementations that decouple weight decay from
#             learning rate. See https://github.com/microsoft/mup/issues/1 for a use case.
#     Outputs:
#         An instance of `impl` with refined parameter groups, each of which has the correctly
#         scaled learning rate according to mup.
#     '''
#     new_param_groups = []
#     for param_group in process_param_groups(params, **kwargs):
#         # For every existing param group, we split into several new groups
#         def new_group():
#             new_g = {k:v for k, v in param_group.items() if k != 'params'}
#             new_g['params'] = []
#             return new_g
#         # The matrix-like weights might need multiple groups since weights
#         # might have different width multipliers
#         matrix_like_p = defaultdict(new_group) # key is width_mult
#         vector_like_p = new_group()
#         for p in param_group['params']:
#             assert hasattr(p, 'infshape'), (
#                 f'A parameter with shape {p.shape} does not have `infshape` attribute. '
#                 'Did you forget to call `mup.set_base_shapes` on the model?')
#             if p.infshape.ninf() == 2:
#                 matrix_like_p[p.infshape.width_mult()]['params'].append(p)
#             elif p.infshape.ninf() > 2:
#                 raise NotImplementedError('more than 2 inf dimensions')
#             else:
#                 vector_like_p['params'].append(p)
#         for width_mult, group in matrix_like_p.items():
#             # Scale learning rate and weight decay accordingly
#             group['lr'] /= width_mult
#             if not decoupled_wd:
#                 group['weight_decay'] *= width_mult
#         new_param_groups.extend(list(matrix_like_p.values()) + [vector_like_p])
#     return impl(new_param_groups, **kwargs)

# def MuAdamW(params, **kwargs):
#     '''AdamW with μP scaling.

#     Note for this to work properly, your model needs to have its base shapes set
#     already using `mup.set_base_shapes`.
#     '''
#     return MuAdam(params, impl=AdamW, **kwargs)


class HackedMuAdamW(composer.optim.DecoupledAdamW):
    # infshape_map_by_id: Dict[str, ]
    params_to_module: Dict[torch.Tensor, nn.Module] = {}
    param_names: Dict[torch.Tensor, List[str]] = {}

    def __init__(self, model: nn.Module, infshapes: Dict[str, mup.shape.InfShape], *args, **kwargs):
        self.model = model
        self.infshapes = infshapes

        super().__init__(*args, **kwargs)
        # print(self.infshapes)

    def add_param_group(self, param_group: dict) -> None:
        self.params_to_module = {}
        self.param_names = {}

        def form_name(m_name: str, p_name: str):
            res = m_name + "." + p_name
            return res.replace("_fsdp_wrapped_module.", "").replace("_fpw_module.", "")

        for m_name, m in self.model.named_modules():
            for p_name, p in m.named_parameters():
                
                self.params_to_module[p] = m
                if isinstance(p, torch.distributed.fsdp.flatten_params_wrapper.FlatParameter):
                    self.param_names[p] = [form_name(m_name, n.param_name) for n in p._param_infos]
                else:
                    self.param_names[p] = [form_name(m_name, p_name)]

        # print(param_names)

        p: torch.Tensor
        for p in param_group["params"]:
            assert p in self.params_to_module, "Trying to add a parameter that's not preregistered"
        return super().add_param_group(param_group)


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # params_with_grad = []
            # grads = []
            # exp_avgs = []
            # exp_avg_sqs = []
            # max_exp_avg_sqs = []
            # state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            eps = group['eps']

            # lr = [] # group['lr']
            # initial_lr = [] # group['initial_lr']
            # weight_decay = [] # group['weight_decay']


            def compute_adamw(p: torch.Tensor, grad: torch.Tensor, label, infshape: mup.shape.InfShape):
                # params_with_grad.append(p)
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                # grads.append(grad)

                state = self.state[label]

                # State initialization
                if len(state) == 0:
                    print('reinitializing!!! ', p.shape)
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)


                # update the steps for each param group update
                state['step'] += 1

                mup_factor: float
                if infshape.ninf() == 1:
                    mup_factor = infshape.width_mult()
                elif infshape.ninf() == 2:
                    mup_factor = 1.0 / infshape.fanin_fanout_mult_ratio()
                elif infshape.ninf() > 2:
                        raise NotImplementedError('more than 2 inf dimensions')
                else:
                    mup_factor = 1.0

                print(mup_factor)

                # lr.append(group['lr'] * mup_factor)
                # initial_lr.append(group['initial_lr'] * mup_factor)
                # weight_decay.append(group['weight_decay'] * mup_factor)


                self.adamw([p],
                        [grad],
                        [state['exp_avg']],
                        [state['exp_avg_sq']],
                        [state['max_exp_avg_sq']] if amsgrad else [],
                        [state['step']],
                        amsgrad=amsgrad,
                        beta1=beta1,
                        beta2=beta2,
                        lr=group['lr'] * mup_factor,
                        initial_lr=group['initial_lr'] * mup_factor,
                        weight_decay=group['weight_decay'] * mup_factor,
                        eps=eps)

            for p in group['params']:
                if p.grad is None:
                    continue

                if isinstance(p, torch.distributed.fsdp.flatten_params_wrapper.FlatParameter):
                    param_views = p.get_param_views()
                    grad_views = (
                        t.view(s)
                        for (t, s) in zip(p.grad.split(p._param_numels), p._param_shapes)
                    )
                    for id, (virtual_param, virtual_grad, info) in enumerate(zip(param_views, grad_views, p._param_infos)):
                        infshape = mup.shape.get_infshape_of_param_name(info.module, info.param_name)

                        # Since virtual view objects are ephemeral we use (flat_param, view_idx) as keys
                        key = (p, id)
                        compute_adamw(virtual_param, virtual_grad, key, infshape)
                else:
                    compute_adamw(p, p.grad, p, mup.shape.get_infshape_of_param_name(self.params_to_module[p], self.param_names[p][0]))

        return loss

# def MuSGD(params, impl=SGD, decoupled_wd=False, **kwargs):
#     '''SGD with μP scaling.

#     Note for this to work properly, your model needs to have its base shapes set
#     already using `mup.set_base_shapes`.
     
#     Inputs:
#         impl: the specific SGD-like optimizer implementation from torch.optim or
#             elsewhere 
#         decoupled_wd: if True, skips the mup scaling for weight decay, which should
#             be used for optimizer implementations that decouple weight decay from
#             learning rate. See https://github.com/microsoft/mup/issues/1 for a use case.
#     Outputs:
#         An instance of `impl` with refined parameter groups, each of which has the correctly
#         scaled learning rate according to mup.
#     '''
#     new_param_groups = []
#     for param_group in process_param_groups(params, **kwargs):
#         # For every existing param group, we split into several new groups
#         def new_group():
#             new_g = {k:v for k, v in param_group.items() if k != 'params'}
#             new_g['params'] = []
#             return new_g
#         # The matrix-like weights might need multiple groups since weights
#         # might have different width multipliers
#         vector_like_p = defaultdict(new_group) # key is width mult
#         matrix_like_p = defaultdict(new_group) # key is fan_in/out ratio
#         fixed_p = new_group()
#         for p in param_group['params']:
#             assert hasattr(p, 'infshape'), (
#                 f'A parameter with shape {p.shape} does not have `infshape` attribute. '
#                 'Did you forget to call `mup.set_base_shapes` on the model?')
#             if p.infshape.ninf() == 1:
#                 vector_like_p[p.infshape.width_mult()]['params'].append(p)
#             elif p.infshape.ninf() == 2:
#                 matrix_like_p[p.infshape.fanin_fanout_mult_ratio()]['params'].append(p)
#             elif p.infshape.ninf() > 2:
#                 raise NotImplementedError('more than 2 inf dimensions')
#             else:
#                 fixed_p['params'].append(p)
#         for width_mult, group in vector_like_p.items():
#             # Scale learning rate and weight decay accordingly
#             group['lr'] *= width_mult
#             if not decoupled_wd:
#                 group['weight_decay'] /= width_mult
#         for shape_ratio, group in matrix_like_p.items():
#             group['lr'] /= shape_ratio
#             if not decoupled_wd:
#                 group['weight_decay'] *= shape_ratio
#         new_param_groups.extend(list(matrix_like_p.values()) + \
#                                 list(vector_like_p.values()) + [fixed_p])
#     return impl(new_param_groups, **kwargs)
