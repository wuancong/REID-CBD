import torch
import torch.nn.functional as F
from torch import nn,autograd

class HM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = grad_outputs.mm(ctx.features)

        for x,y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1 - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None

def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs,indexes,features,torch.Tensor([momentum]).to(inputs.device))

class Memory(nn.Module):
    def __init__(self,num_features, num_samples, temp=0.05, momentum=0.2):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features',torch.zeros(num_samples,num_features))
        self.register_buffer('label',torch.zeros(num_samples).long())

    def forward(self, inputs, indexes):
        inputs = hm(inputs, indexes, self.fetures, self.momentum)
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, eplison=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + eplison
            return (masked_exps/masked_sums)
        targets = self.labels[indexes].clone()
        labels = self.labeds.clone()

        sim = torch.zeros(labels.max() + 1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max() + 1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples, 1).float().cuda())
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)


