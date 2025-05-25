import torch
from torch.autograd import Function

class ExpXPlusCosY(Function):
    @staticmethod
    def forward(ctx, x, y):
        exp_x = torch.exp(x)
        cos_y = torch.cos(y)
        result = exp_x + cos_y
        ctx.save_for_backward(x, y, exp_x, cos_y)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, exp_x, cos_y = ctx.saved_tensors
        grad_x = grad_output * exp_x
        grad_y = grad_output * (-torch.sin(y))
        return grad_x, grad_y

def test_autograd_function():
    x = torch.randn(3, requires_grad=True)
    y = torch.randn(3, requires_grad=True)

    our_output = ExpXPlusCosY.apply(x, y)
    our_output.sum().backward()
    our_grad_x = x.grad.clone()
    our_grad_y = y.grad.clone()
    
    x.grad = None
    y.grad = None
    
    ref_output = torch.exp(x) + torch.cos(y)
    ref_output.sum().backward()
    ref_grad_x = x.grad.clone()
    ref_grad_y = y.grad.clone()
    
    print("Forward diff:", torch.max(torch.abs(our_output - ref_output)).item())
    print("Grad x diff:", torch.max(torch.abs(our_grad_x - ref_grad_x)).item())
    print("Grad y diff:", torch.max(torch.abs(our_grad_y - ref_grad_y)).item())

if __name__ == "__main__":
    test_autograd_function()


# Forward diff: 0.0
# Grad x diff: 0.0
# Grad y diff: 0.0