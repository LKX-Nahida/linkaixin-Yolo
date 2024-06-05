from models.common import *

NET_CONFIG = {
    "blocks2":
    # k, in_c, out_c, s, use_se
        [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}
BLOCK_LIST = ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]


def make_divisible_LC(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return (self.relu6(x + 3)) / 6


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            HardSigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparable(nn.Module):
    def __init__(self, inp, oup, dw_size, stride, use_se=False):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self.stride = stride
        self.inp = inp
        self.oup = oup
        self.dw_size = dw_size
        self.dw_sp = nn.Sequential(
            nn.Conv2d(self.inp, self.inp, kernel_size=self.dw_size, stride=self.stride,
                      padding=autopad(self.dw_size, None), groups=self.inp, bias=False),
            nn.BatchNorm2d(self.inp),
            HardSwish(),

            nn.Conv2d(self.inp, self.oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.oup),
            HardSwish(),
        )
        self.se = SELayer(self.oup)

    def forward(self, x):
        x = self.dw_sp(x)
        if self.use_se:
            x = self.se(x)
        return x


class PPLC_Conv(nn.Module):
    def __init__(self, scale):
        super(PPLC_Conv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(3, out_channels=make_divisible_LC(16 * self.scale),
                              kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class PPLC_Block(nn.Module):
    def __init__(self, scale, block_num):
        super(PPLC_Block, self).__init__()
        self.scale = scale
        self.block_num = BLOCK_LIST[block_num]
        self.block = nn.Sequential(*[
            DepthwiseSeparable(inp=make_divisible_LC(in_c * self.scale),
                               oup=make_divisible_LC(out_c * self.scale),
                               dw_size=k, stride=s, use_se=use_se)
            for i, (k, in_c, out_c, s, use_se) in enumerate(NET_CONFIG[self.block_num])
        ])

    def forward(self, x):
        return self.block(x)
