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

import mupx
import mupx.shape
import mupx.init
import mupx.optim
from mupx.layer import get_infshape_of_param_name

import copy

import composer.optim

import torch.distributed.fsdp.flatten_params_wrapper

class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.linear = nn.Linear(10, 10, bias=False)

        self.act1 = nn.ReLU()

        self.readout = mupx.MuSharedReadout(self.linear.weight)
        # self.readout = mup.MuReadout(10, 5)
    
    def forward(self, x):
        return self.readout(self.act1(self.linear(x)))

    def initia(self):
        mupx.init.xavier_normal_(self.linear, self.linear.weight)
        # mup.init.xavier_normal_(self.readout, self.readout.weight)

def expand_params(model: nn.Module):
    params_with_shapes: List[Tuple[torch.Tensor, mupx.InfShape]] = []
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


if __name__ == "__main__":
    print("wawawa")

    a = A().cuda()

    # print(a.linear.weight)

    # a.initialize()
    a: A = mupx.set_base_shapes(a, a)

    a.initia()

    infshapes = mupx.shape.get_infshapes(a)

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

    # opt = mup.optim.MuAdamW(b.parameters(), lr=1e-3, model=b)

# 
    opt = mupx.optim.HackedMuAdamW(b, infshapes, b.parameters(), lr=1e-3)

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
