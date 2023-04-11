import torch
EPS = 1e-5

def weighted_mean(h, a):
    return torch.sum(h*a, dim=1)

def weighted_max(h, a):
    return torch.max(h*a,dim=1)[0]

def weighted_min(h, a):
    return torch.min(h*a,dim=1)[0]

def weighted_var(h, a):
    mean = weighted_mean(h,a)
    var = torch.sum(torch.square(h-mean.unsqueeze(1).expand_as(h)) * a,dim=1)
    return var

def weighted_std(h, a):
    return torch.sqrt(weighted_var(h,a)+EPS)