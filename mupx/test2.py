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
    def __init__(self, readout_cls, init_xavier_normal):
        super(A, self).__init__()
        self.linear = nn.Linear(10, 10, bias=False)

        self.act1 = nn.ReLU()

        self.readout = readout_cls(self.linear.weight, bias=False)

        self.init_xavier_normal = init_xavier_normal
        # self.readout = mup.MuReadout(10, 5)
    
    def forward(self, x):
        return self.readout(self.act1(self.linear(x)))

    def initia(self):
        self.init_xavier_normal(self.linear, self.linear.weight)
        # mupx.init.xavier_normal_(self.linear, self.linear.weight)
        # mup.init.xavier_normal_(self.readout, self.readout.weight)

    # def _

def test_my_mup():
    a = A(mupx.MuSharedReadout, mupx.init.xavier_normal_).cuda()
    a: A = mupx.set_base_shapes(a, a)

    rand_inp = torch.randn((1, 10), dtype=torch.float32).cuda()
    rand_out = torch.randn((1, 10), dtype=torch.float32).cuda()

    infshapes = mupx.get_infshapes(a)

    print(infshapes)

    b = FSDP(a)


    opt = mupx.optim.HackedMuAdamW(b, infshapes, b.parameters(), lr=1e-2)

    # for p in b.parameters():
    #     print(p)
    
    # for i in range(3):
    #     opt.zero_grad()
    #     output = b(rand_inp)
    #     loss = torch.nn.MSELoss()(output, rand_out)
    #     loss.backward()
    #     opt.step()

    # print("")
    # print("")
    # print("")
    # print("")

    # for p in b.parameters():
    #     print(p)


def test_other_mup():
    import mup
    import mup.init
    import mup.optim

    def xav_wrap(_, *args):
        mup.init.xavier_normal_(*args)

    a = A(mup.MuSharedReadout, xav_wrap).cuda()
    a: A = mup.set_base_shapes(a, a)

    print(mup.get_infshapes(a))

    rand_inp = torch.randn((1, 10), dtype=torch.float32).cuda()
    rand_out = torch.randn((1, 10), dtype=torch.float32).cuda()

    # infshapes = mup.get_infshapes(a)

    # b = FSDP(a)

    opt = mup.optim.MuAdamW(a.parameters(), lr=1e-2)

    # opt = mupx.optim.HackedMuAdamW(b, infshapes, b.parameters(), lr=1e-3)

    # for p in a.parameters():
    #     print(p)
    
    # for i in range(3):
    #     opt.zero_grad()
    #     output = a(rand_inp)
    #     loss = torch.nn.MSELoss()(output, rand_out)
    #     loss.backward()
    #     opt.step()

    # print("")
    # print("")
    # print("")
    # print("")

    # for p in a.parameters():
    #     print(p)
    


if __name__ == "__main__":
    torch.manual_seed(0)
    dist.initialize_dist(composer.utils.get_device('gpu'), 500)
    test_my_mup()

    print()
    print("other mup:")
    print()

    test_other_mup()