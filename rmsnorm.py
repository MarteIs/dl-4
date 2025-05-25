import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def test_rmsnorm():
    dim = 512
    batch_size = 10
    x = torch.randn(batch_size, dim)
    rms_norm = RMSNorm(dim)
    our_output = rms_norm(x)
    
    try:
        official_rmsnorm = nn.RMSNorm(dim)
        official_output = official_rmsnorm(x)
        
        print("Forward diff:", torch.max(torch.abs(our_output - official_output)).item())
        
        our_output.sum().backward()
        official_output.sum().backward()
        
        print("Weight grad diff:", torch.max(torch.abs(rms_norm.weight.grad - official_rmsnorm.weight.grad)).item())
    except AttributeError:
        print("Output mean:", our_output.mean().item(), "std:", our_output.std().item())

if __name__ == "__main__":
    test_rmsnorm()


# Forward diff: 2.1457672119140625e-06
# Weight grad diff: 4.76837158203125e-06