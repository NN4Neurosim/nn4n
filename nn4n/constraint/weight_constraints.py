def weight_check(W):
    # must be torch tensor
    assert isinstance(W, torch.Tensor), "W must be a torch tensor"

def frobenius_norm(W, avg=True):
    weight_check(W)
    return T.sqrt(T.sum(W**2))/W.numel() if avg else T.sqrt(T.sum(W**2))

def l1_norm(W, avg=True):
    weight_check(W)
    return T.sum(T.abs(W))/W.numel() if avg else T.sum(T.abs(W))
