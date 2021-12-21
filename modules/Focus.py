import torch.nn as nn
from torch.nn import Conv2d as Conv
import torch


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, kernel_size=k, stride=s, padding=p, groups=g, bias=act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

if __name__ == '__main__':
    data = torch.rand(1, 3, 224, 224)
    f = Focus(3, 12)
    out = f(data)
    print(out.shape)