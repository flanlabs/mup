from typing import List, Tuple
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
import mup.init
from mup.layer import get_infshape_of_param_name

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

# class HackedAdamW(torch.optim.AdamW):

#     infshape_map_by_id: 

#     def __init__(self, model: nn.Module, *args, **kwargs):
#         self.model = model


#     def add_param_group(self, param_group: dict) -> None:
#         p: torch.Tensor
#         for p in param_group["params"]:
#             if p.
#         return super().add_param_group(param_group)

if __name__ == "__main__":
    print("wawawa")

    a = A()

    # print(a.linear.weight)

    # a.initialize()
    a: A = mup.set_base_shapes(a, a)

    a.initia()

    # print([b for (_, b) in expand_params(a)])

    rand_inp = torch.randn((1, 10), dtype=torch.float32)
    rand_out = torch.randn((1, 5), dtype=torch.float32)

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

    opt = torch.optim.SGD([x for (x, _) in expand_params(b)], lr=1e-3)


    torch.optim.AdamW

    opt.zero_grad()
    output = b(rand_inp)
    loss = torch.nn.MSELoss()(output, rand_out)
    loss.backward()
    opt.step()

    # print([z for (_, z) in expand_params(b)])
    # print(b)

    # for p in b.parameters():
    #     assert isinstance(p, torch.distributed.fsdp.flatten_params_wrapper.FlatParameter)
    #     print(p._param_infos)


    # print(a.parameters())

