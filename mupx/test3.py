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

from mupx.optim import ViewWithGrad

import copy

import composer.optim

from torch.distributed.fsdp.flatten_params_wrapper import FlatParameter

import mup
import mup.optim
import mup.init

class A(nn.Module):
    def __init__(self, readout_cls, init_xavier_normal):
        super(A, self).__init__()
        self.linear = nn.Linear(10, 10, bias=False)

        self.linear2 = nn.Sequential(*[nn.Linear(10, 10, bias=False) for _ in range(10)])

        self.act1 = nn.ReLU()

        self.readout = readout_cls(self.linear.weight, bias=False)

        self.init_xavier_normal = init_xavier_normal
        # self.readout = mup.MuReadout(10, 5)
    
    def forward(self, x):
        r1 = self.linear(x)
        r2 = self.linear2(r1)
        r3 = self.act1(r2)
        return self.readout(r3)
        # return self.readout(self.act1(self.linear2(self.linear(x))))

    def initia(self):
        self.init_xavier_normal(self.linear, self.linear.weight)
        # mupx.init.xavier_normal_(self.linear, self.linear.weight)
        # mup.init.xavier_normal_(self.readout, self.readout.weight)


def form_name(m_name: str, p_name: str):
    res = m_name + "." + p_name
    return res.replace("_fsdp_wrapped_module.", "").replace("_fpw_module.", "")


if __name__ == "__main__":
    torch.manual_seed(0)
    dist.initialize_dist(composer.utils.get_device('gpu'), 500)
    
    a = A(mup.MuSharedReadout, (lambda x, *args: mup.init.xavier_normal_(*args))).cuda()
    a: A = mup.set_base_shapes(a, a, rescale_params=False)

    infshapes = {a: b.infshape for (a, b) in a.named_parameters()}

    print(infshapes)

    rand_inp = torch.randn((1, 10), dtype=torch.float32).cuda()
    rand_out = torch.randn((1, 10), dtype=torch.float32).cuda()

    # infshapes = mupx.get_infshapes(a)

    # print(infshapes)

    b = FSDP(a)

    def compute_views(p: FlatParameter) -> List[ViewWithGrad]:
        res: List[ViewWithGrad]= []

        if not p._is_sharded:
            for info, offset, size in zip(p._param_infos, p._param_offsets, p._param_numels):
                res.append(ViewWithGrad(p, info, offset[0], size))
            return res

        # if sharded, we need some more math
        accum = 0
        for info, (start, end) in zip(p._param_infos[p._offset_to_slice()], p._sharded_param_offsets):
            size = end - start
            assert size >= 0
            res.append(ViewWithGrad(p, info, accum, size))
            accum += size
        
        assert accum <= p.numel(), f"{accum=}, {p.numel()=}"
        return res


    logical_params = []
    for p in b.parameters():
        vs = compute_views(p)
        
        for v in vs:
            prev_name = form_name(v.info.module_name, v.info.param_name)
            # print(prev_name)

            v.infshape = infshapes[prev_name]

        logical_params.extend(vs)
        # print(vs[1].add(vs[2]))
        # print(p._param_infos)
        # print(p._offset_to_slice())
        # # print(p.get_param_views())
        # print(p._sharded_param_offsets)


    
    opt = mup.optim.MuAdamW(logical_params, lr=1e-2)

    # for p in b.parameters():
    #     print(p)
    
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

    # print(b)
