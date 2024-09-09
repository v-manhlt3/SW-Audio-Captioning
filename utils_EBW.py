import torch
from torch.nn.functional import pad
from ot.lp import wasserstein_1d
import ot

def SW(X, Y, a, b, L, p=2):
    d = X.shape[1]
    X = X.view(X.shape[0], -1)
    Y = Y.view(Y.shape[0], -1)
    thetas = torch.randn(d, L, device=X.device)
    thetas = thetas/torch.sqrt(torch.sum(thetas**2, dim=0, keepdim=True))
    X_prod = torch.matmul(X, thetas)
    Y_prod = torch.matmul(Y, thetas)
    return torch.mean(one_dimensional_Wasserstein(X_prod, Y_prod, a, b, p))**(1./p)

def kernel_SW_org(X, Y, a, b, L, p=2, gamma=1):
    d = X.shape[1]
    X = X.view(X.shape[0], -1)
    Y = Y.view(Y.shape[0], -1)
    thetas = torch.randn(d, L, device=X.device)
    thetas = thetas/torch.sqrt(torch.sum(thetas**2, dim=0, keepdim=True))
    X_prod = torch.matmul(X, thetas)
    Y_prod = torch.matmul(Y, thetas)
    sw_dist = one_dimensional_Wasserstein(X_prod, Y_prod, a, b, p)
    kernel_dist = torch.exp(-gamma*sw_dist)
    return torch.mean(kernel_dist)

def kernel_SW(X, Y, a, b, L, p=2, gamma=1):
    d = X.shape[1]
    thetas = ot.sliced.get_random_projections(d, L)
    thetas = torch.tensor(thetas).float().to(X.device)
    # print("Thetas shape: ", thetas.shape)
    # print("X shape: ", X.shape)
    # print("*"*90)
    X_prod = X@thetas
    Y_prod = Y@thetas
    sw_dist = wasserstein_1d(X_prod, Y_prod, a,b,p)
    kernel_dist = torch.exp(-gamma*sw_dist)
    return torch.mean(kernel_dist)
    # return torch.mean(one_dimensional_Wasserstein(X_prod, Y_prod, a, b, p))**(1./p)


def HybridEBSW(X, Y, a, b, L, temp=1, p=2,alpha=0.5):
    #X =[X1,X2], X1 \in R^{n,1024} X2 \in R^{n,d}, Y =[Y1,Y2], Y1 \in R^{m,1024} Y2 \in R^{m,d}
    d = X.shape[1]
    X = X.view(X.shape[0], -1)
    Y = Y.view(Y.shape[0], -1)
    thetas = torch.randn(d, L, device=X.device)
    
    thetas = thetas/torch.sqrt(torch.sum(thetas**2, dim=0, keepdim=True))
    prefix_weight = torch.ones(1, 1024).to(X.device)*0.7
    suffix_weight = torch.ones(1, 1024).to(X.device)*0.3
    weight_vec = torch.concat([prefix_weight, suffix_weight], dim=-1)

    # X_prod = torch.matmul(X, thetas)
    # Y_prod = torch.matmul(Y, thetas)

    X_prod = torch.matmul(X*weight_vec, thetas)
    Y_prod = torch.matmul(Y*weight_vec, thetas)
    sws= one_dimensional_Wasserstein(X_prod, Y_prod, a, b, p)
    # CO the cong them sws
    # print("sum shape: ", torch.sum(torch.abs(thetas[:1024, :]),dim=0, keepdim=True).shape)
    # print("second sum: ", torch.sum(torch.abs(thetas[1024:, :]),dim=0, keepdim=True).shape)
    # print("sws shape: ", sws.shape)
    # weights = alpha*torch.sum(torch.abs(thetas[:1024, :]),dim=0, keepdim=True)+ (1-alpha)*torch.sum(torch.abs(thetas[1024:, :]),dim=0, keepdim=True)
    # # weights = weights/torch.sum(weights)
    # weights = torch.softmax(weights/temp,dim=-1)
    # print("weight: ", sws.shape)
    # return torch.sum(sws*weights)
    return torch.mean(sws)


def EBSW(X, Y, a, b, L, temp=1, p=2):
    d = X.shape[1]
    X = X.view(X.shape[0], -1)
    Y = Y.view(Y.shape[0], -1)
    thetas = torch.randn(d, L, device=X.device)
    thetas = thetas/torch.sqrt(torch.sum(thetas**2, dim=0, keepdim=True))
    X_prod = torch.matmul(X, thetas)
    Y_prod = torch.matmul(Y, thetas)
    sws= one_dimensional_Wasserstein(X_prod, Y_prod, a, b, p)
    weights = torch.softmax(sws/temp,dim=-1)
    return torch.sum(sws*weights)


def quantile_function(qs, cws, xs):
    n = xs.shape[0]
    cws = cws.T.contiguous()
    qs = qs.T.contiguous()
    idx = torch.searchsorted(cws, qs, right=False).T
    return torch.gather(xs, 0, torch.clamp(idx, 0, n - 1))


def one_dimensional_Wasserstein(u_values, v_values, u_weights=None, v_weights=None, p=2):
    n = u_values.shape[0]
    m = v_values.shape[0]

    if u_weights is None:
        u_weights = torch.full(u_values.shape, 1. / n,
                               dtype=u_values.dtype, device=u_values.device)
    elif u_weights.ndim != u_values.ndim:
        u_weights = torch.repeat_interleave(
            u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = torch.full(v_values.shape, 1. / m,
                               dtype=v_values.dtype, device=v_values.device)
    elif v_weights.ndim != v_values.ndim:
        v_weights = torch.repeat_interleave(
            v_weights[..., None], v_values.shape[-1], -1)

    u_sorter = torch.sort(u_values, 0)[1]
    u_values = torch.gather(u_values, 0, u_sorter)

    v_sorter = torch.sort(v_values, 0)[1]
    v_values = torch.gather(v_values, 0, v_sorter)

    u_weights = torch.gather(u_weights, 0, u_sorter)
    v_weights = torch.gather(v_weights, 0, v_sorter)

    u_cumweights = torch.cumsum(u_weights, 0)
    v_cumweights = torch.cumsum(v_weights, 0)

    qs = torch.sort(torch.cat((u_cumweights, v_cumweights), 0), 0)[0]
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)

    pad_width = [(1, 0)] + (qs.ndim - 1) * [(0, 0)]
    how_pad = tuple(element for tupl in pad_width[::-1] for element in tupl)
    qs = pad(qs, how_pad)

    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = torch.abs(u_quantiles - v_quantiles)
    return torch.sum(delta * torch.pow(diff_quantiles, p), dim=0, keepdim=True)

if __name__ == "__main__":

    X = torch.rand(100, 1124)
    a = torch.ones(X.shape[0])/X.shape[0]
    Y = torch.rand(10, 1124)
    b = torch.ones(Y.shape[0])/Y.shape[0]

    dis = HybridEBSW(X, Y, a, b, 10)
    print("distance: ", dis)
