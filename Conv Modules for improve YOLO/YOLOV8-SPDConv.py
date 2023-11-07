import torch
import torch.nn as nn

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


if __name__ == '__main__':

    input = torch.randn(1, 128, 8, 8)
    dsconv = SPDConv()
    output = dsconv(input)
    print(output.shape)
