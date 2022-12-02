
# Note: this pretends to be a normal Tensor but instead is a view into a FlatParameter.
# This is very ugly and might break in some use cases, please beware.
class ViewIntoFlatParam(object):
    # @property
    # def __class__(self):
    #     return torch.Tensor

    def __init__(self, p: FlatParameter, idx: int):
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "idx", idx)
        object.__setattr__(self, "view", p.split_with_sizes(p._param_numels)[idx].view(p._param_shapes[idx]))
        # self.p = p
        # self.idx = idx
        # self.shape = p._param_shapes[idx]
        # self.numels = p._param_numels[idx]
        # self.info = p._param_infos[idx]
        # self.view = self.p.split_with_sizes(p._param_numels)[idx].view(p._param_shapes[idx])

    # @property
    # def is_leaf(self):
    #     return self.p.is_leaf

    # @property
    # def grad(self):
    #     if self.p.grad == None: return None
    #     grad_view = self.p.grad.split_with_sizes(self.p._param_numels)[self.idx].view(self.p._param_shapes[self.idx])
    #     return grad_view

    def __getattribute__(self, name):
        if name == "is_leaf":
            return object.__getattribute__(self, "p").is_leaf
        if name == "info":
            p = object.__getattribute__(self, "p")
            idx = object.__getattribute__(self, "idx")
            return p._param_infos[idx]
        if name == "grad":
            p = object.__getattribute__(self, "p")
            if p.grad is None:
                return None
            idx = object.__getattribute__(self, "idx")
            grad_view = p.grad.split_with_sizes(p._param_numels)[idx].view(p._param_shapes[idx])
            return grad_view

        if name == "infshape":
            return object.__getattribute__(self, "infshape")



        return object.__getattribute__(object.__getattribute__(self, "view"), name)

# class ViewIntoFlatParam2(object):
#     def __getattribute__(self, __name: str) -> Any:
#         # def __init__(self, p)
#         def __init__(self, p: FlatParameter, idx: int):
#             self.p = p
#             self.idx = idx
#             # self.shape = p._param_shapes[idx]
#             # self.numels = p._param_numels[idx]
#             self.info = p._param_infos[idx]
#             self.view = self.p.split_with_sizes(p._param_numels)[idx].view(p._param_shapes[idx])



    # @property
    # def is_leaf(self):
    #     return self.p.is_leaf

    # @property
    # def grad(self):
    #     if self.p.grad == None: return None
    #     grad_view = self.p.grad.split_with_sizes(self.p._param_numels)[self.idx].view(self.p._param_shapes[self.idx])
    #     return grad_view

    # def __getattr__(self, name):
    #     return getattr(self.view, name)

def wrap_flatparams(params: List[torch.Tensor]) -> List[torch.Tensor]:
    new_params = []
    for p in params:
        if isinstance(p, FlatParameter):
            new_params.extend([ViewIntoFlatParam(p, i) for i in range(len(p._param_infos))])
        else:
            new_params.append(p)

    return new_params


def attach_infshapes_to_params(params: List[torch.Tensor]) -> List[torch.Tensor]:
    new_params = []
    for p in params:
        if isinstance(p, FlatParameter):
            new_params.extend([ViewIntoFlatParam(p, i) for i in range(len(p._param_infos))])
        else:
            new_params.append(p)

    return new_params

def process_param_groups(params, model: nn.Module, **kwargs):
    param_groups = list(params)
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]

    cache_infshapes = mup.shape.get_infshapes(model)
    cache_params_to_module = {}
    cache_param_names = {}

    def form_name(m_name: str, p_name: str):
        res = m_name + "." + p_name
        return res.replace("_fsdp_wrapped_module.", "").replace("_fpw_module.", "")

    # cache some model info

    for m_name, m in model.named_modules():
        for p_name, p in m.named_parameters(recurse=False):
            
            cache_params_to_module[p] = m
            if isinstance(p, FlatParameter):
                cache_param_names[p] = [form_name(m_name, n.param_name) for n in p._param_infos]
            else:
                cache_param_names[p] = [form_name(m_name, p_name)]

    # do the actual work

    for param_group in param_groups:
        param_group['params'] = wrap_flatparams(param_group['params'])

        for p in param_group['params']:
            if isinstance(p, ViewIntoFlatParam):
                p.infshape = mup.shape.get_infshape_of_param_name(p.info.module, p.info.param_name)
            else:
                p.infshape = cache_infshapes[cache_param_names[p]]

        # for p in param_group['params']:
        #     assert hasattr(p, 'infshape')
        if 'lr' not in param_group:
            param_group['lr'] = kwargs['lr']
        if 'weight_decay' not in param_group:
            param_group['weight_decay'] = kwargs.get('weight_decay', 0.)
    return param_groups
