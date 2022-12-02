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
from torch.distributed.fsdp.flatten_params_wrapper import FlatParameter
import composer.optim
import mupx.shape

class HackedMuAdamW(composer.optim.DecoupledAdamW):
    # infshape_map_by_id: Dict[str, ]
    params_to_module: Dict[torch.Tensor, nn.Module] = {}
    param_names: Dict[torch.Tensor, List[str]] = {}

    def __init__(self, model: nn.Module, infshapes: Dict[str, mupx.shape.InfShape], *args, **kwargs):
        self.model = model
        self.infshapes = infshapes

        super().__init__(*args, **kwargs)

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


            def compute_adamw(p: torch.Tensor, grad: torch.Tensor, label, infshape: mupx.shape.InfShape):
                # params_with_grad.append(p)
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                # grads.append(grad)

                state = self.state[label]

                # State initialization
                if len(state) == 0:
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
                        infshape = mupx.shape.get_infshape_of_param_name(info.module, info.param_name)

                        # Since virtual view objects are ephemeral we use (flat_param, view_idx) as keys
                        key = (p, id)
                        compute_adamw(virtual_param, virtual_grad, key, infshape)
                else:
                    compute_adamw(p, p.grad, p, mupx.shape.get_infshape_of_param_name(self.params_to_module[p], self.param_names[p][0].split(".")[-1]))

        return loss
