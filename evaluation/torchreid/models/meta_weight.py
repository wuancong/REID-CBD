import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.01)



class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        #self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight)

    def named_leaves(self):
        return [('weight', self.weight)]

class VNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden,bias=False)
        self.linear1.apply(weights_init_classifier)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden, output,bias=False)
        self.linear2.apply(weights_init_classifier)



    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

class Sample_net_anchor(MetaModule):
    def __init__(self, inplane, hidden, plane):
        super(Sample_net_anchor,self).__init__()
        #self.linear1 = nn.Linear(inplane, hidden, bias=False)
        self.linear1 = MetaLinear(inplane, hidden, bias=False)
        self.linear1.apply(weights_init_classifier)
     
        self.relu = nn.ReLU(inplace=True)
        #self.bn = nn.BatchNorm1d(hidden)

        #self.linear2 = nn.Linear(hidden, plane, bias=False)
        self.linear2 = MetaLinear(hidden, plane, bias=False)
        self.linear2.apply(weights_init_classifier)
     

    def forward(self, x):
        x = self.linear1(x)
        #x = self.bn(x)
        x = self.relu(x)

        x = self.linear2(x)

        x = self.relu(x)

        return x


class Sample_net_pair(MetaModule):
    def __init__(self,inplane, hidden, plane):
        super(Sample_net_pair, self).__init__()
        #self.linear1 = nn.Linear(inplane,hidden,bias=False)
        self.linear1 = MetaLinear(inplane, hidden, bias=False)
        self.linear1.apply(weights_init_classifier)
        self.relu = nn.ReLU(inplace=True)
        #self.bn = nn.BatchNorm1d(hidden)

        #self.linear2 = nn.Linear(hidden, plane, bias=False)
        self.linear2 = MetaLinear(hidden, plane, bias=False)
        self.linear2.apply(weights_init_classifier)
   

    def forward(self, x):
        x = self.linear1(x)
        #x = self.bn(x)
        x = self.relu(x)

        x = self.linear2(x)

        x = self.relu(x)

        return x

