from typing import Dict, List, Tuple
import torch.nn as nn
import torch

from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
import torch.distributed
import torch.random
import os
import composer
from composer.utils import dist
import composer.utils

import mup
import mup.shape
import mup.init
from mup.layer import get_infshape_of_param_name

import copy

import composer.optim

import torch.distributed.fsdp.flatten_params_wrapper

class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.linear = nn.Linear(10, 10, bias=False)

        self.act1 = nn.ReLU()

        self.readout = mup.MuSharedReadout(self.linear.weight)
        # self.readout = mup.MuReadout(10, 5)
    
    def forward(self, x):
        return self.readout(self.act1(self.linear(x)))

    def initia(self):
        mup.init.xavier_normal_(self.linear, self.linear.weight)
        # mup.init.xavier_normal_(self.readout, self.readout.weight)

def expand_params(model: nn.Module):
    params_with_shapes: List[Tuple[torch.Tensor, mup.InfShape]] = []
    for m_name, m in model.named_modules():
        for p_name, p in m.named_parameters(recurse=False):
            if not isinstance(p, torch.distributed.fsdp.flatten_params_wrapper.FlatParameter):
                params_with_shapes.append((p, get_infshape_of_param_name(m, p_name)))
                continue
            
            # Otherwise, we have a FlatParameter. What now?
            # orig_params = p._param_infos
            # p.get_param_views()
            for op_info, op_view in zip(p._param_infos, p.get_param_views()):
                # print(op.param_name)
                infshape = get_infshape_of_param_name(op_info.module, op_info.param_name)
                params_with_shapes.append((op_view, infshape))
            # raise NotImplementedError("asdf")

    return params_with_shapes

class HackedAdamW(composer.optim.DecoupledAdamW):
    # infshape_map_by_id: Dict[str, ]
    params_to_module: Dict[torch.Tensor, nn.Module] = {}
    param_names: Dict[torch.Tensor, List[str]] = {}

    def __init__(self, model: nn.Module, infshapes: Dict[str, mup.InfShape], *args, **kwargs):
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
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            initial_lr = group['initial_lr']
            weight_decay = group['weight_decay']


            def add_task(p: torch.Tensor, grad: torch.Tensor, label, infshape: mup.InfShape):
                params_with_grad.append(p)
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(grad)

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

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            for p in group['params']:
                if p.grad is None:
                    continue

                if isinstance(p, torch.distributed.fsdp.flatten_params_wrapper.FlatParameter):
                    param_views = p.get_param_views()
                    grad_views = (
                        t.view(s)
                        for (t, s) in zip(p.grad.split(p._param_numels), p._param_shapes)
                    )
                    for id, (virtual_param, virtual_grad) in enumerate(zip(param_views, grad_views)):
                        # Since virtual view objects are ephemeral we use (flat_param, view_idx) as keys
                        add_task(virtual_param, virtual_grad, (p, id))
                else:
                    add_task(p, p.grad, p, mup.get_infshape_of_param_name(self.params_to_module[p], self.param_names[p][0]))

                

            self.adamw(params_with_grad,
                       grads,
                       exp_avgs,
                       exp_avg_sqs,
                       max_exp_avg_sqs,
                       state_steps,
                       amsgrad=amsgrad,
                       beta1=beta1,
                       beta2=beta2,
                       lr=lr,
                       initial_lr=initial_lr,
                       weight_decay=weight_decay,
                       eps=eps)

        return loss

if __name__ == "__main__":
    print("wawawa")

    a = A().cuda()

    # print(a.linear.weight)

    # a.initialize()
    a: A = mup.set_base_shapes(a, a)

    a.initia()

    infshapes = mup.shape.get_infshapes(a)

    # print([b for (_, b) in expand_params(a)])

    rand_inp = torch.randn((1, 10), dtype=torch.float32).cuda()
    rand_out = torch.randn((1, 10), dtype=torch.float32).cuda()

    # forward pass works!
    # print(a(rand_inp))


    
    # print(list(a.linear.named_parameters(recurse=False)))
    

    dist.initialize_dist(composer.utils.get_device('gpu'), 500)

    # os.environ["RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "1234"
    # # os.

    # torch.distributed.init_process_group('gloo')
    def sharding_strategy(module: nn.Module, recurse: bool, unwrapped_params: int):
        return True

    b = FSDP(a, auto_wrap_policy=sharding_strategy)

    opt = HackedAdamW(b, infshapes, b.parameters(), lr=1e-3)

    # opt.add_param_group({"params": b.parameters()})

    # mup.optim

    # torch.optim.AdamW

    for p in b.parameters():
        print(p)
    
    for i in range(3):
        opt.zero_grad()
        output = b(rand_inp)
        loss = torch.nn.MSELoss()(output, rand_out)
        loss.backward()
        opt.step()

    print("")
    print("")
    print("")
    print("")

    for p in b.parameters():
        print(p)
    


    # print([z for (_, z) in expand_params(b)])
    # print(b)

    # for p in b.parameters():
    #     assert isinstance(p, torch.distributed.fsdp.flatten_params_wrapper.FlatParameter)
    #     print(p._param_infos)
    # composer.Trainer

    # print(a.parameters())
