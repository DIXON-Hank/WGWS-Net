import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import matplotlib.pyplot as plt

def feature_save1(tensor, name):
    # tensor = torchvision.utils.make_grid(tensor.transpose(1,0))
    # tensor = torch.mean(tensor,dim=1).repeat(3,1,1)
    if not os.path.exists(str(name)):
        os.makedirs(str(name))
    for i in range(tensor.shape[1]):
        inp = tensor[:,i,:,:].detach().cpu().numpy().transpose(1,2,0)
        inp = np.clip(np.abs(inp),0,1)
        inp = (inp-np.min(inp))/(np.max(inp)-np.min(inp))
        inp = np.squeeze(inp)
        plt.figure()
        plt.imshow(inp)
        plt.savefig(str(name) + '/' + str(i) + '.png')

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias= True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        #layers = list()
        self.transpose= transpose
        if self.transpose:
            padding = kernel_size // 2 -1
            self.layer = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)
        else:
            self.layer = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)

        self.relu= relu
        if self.relu:
            self.act = nn.GELU()

    def forward(self, x):
        if self.relu:
            # if self.transpose:
            #     return self.act(self.layer(x))
            # else:
            return self.act(self.layer(x))
        else:
            # if self.transpose:
            #     return self.layer(x)
            # else:
            return self.layer(x)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = BasicConv(in_channel=Cin, out_channel=G, kernel_size=kSize, stride=1, relu=True)
    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDBlock(nn.Module):
    def __init__(self, in_channel, out_channel, nConvLayers=3):
        super(RDBlock, self).__init__()
        G0 = in_channel
        G = in_channel
        C = nConvLayers

        self.conv0 = RDB_Conv(G0 , G)
        self.conv1 = RDB_Conv(G0 + 1 * G , G)
        self.conv2 = RDB_Conv(G0 + 2 * G , G)
        # Local Feature Fusion
        self.LFF = BasicConv(in_channel=G0 + C * G, out_channel=out_channel, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.LFF(out) + x
        return out


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        self.layer1 = RDBlock(out_channel, out_channel)
        self.layer2 = RDBlock(out_channel, out_channel)
        self.layer3 = RDBlock(out_channel, out_channel)
        self.layer4 = RDBlock(out_channel, out_channel)
        self.layer5 = RDBlock(out_channel, out_channel)
        self.layer6 = RDBlock(out_channel, out_channel)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        # layers = [RDBlock(channel, channel) for _ in range(num_res)]
        # self.layers = nn.Sequential(*layers)
        self.layer1 = RDBlock(channel, channel)
        self.layer2 = RDBlock(channel, channel)
        self.layer3 = RDBlock(channel, channel)
        self.layer4 = RDBlock(channel, channel)
        self.layer5 = RDBlock(channel, channel)
        self.layer6 = RDBlock(channel, channel)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()

        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.layer1 = BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True)
        self.layer2 = BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True)
        self.layer3 = BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True)
        self.layer4 = BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        x = torch.cat([x,out], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: Decoder feature map, x: Encoder feature map
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    
class UNet(nn.Module):
    def __init__(self, base_channel=24, num_res=6):
        super(UNet, self).__init__()

        base_channel = base_channel

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        # Add Attention Gates for each skip connection
        self.AttentionGates = nn.ModuleList([
            AttentionGate(F_g=base_channel * 2, F_l=base_channel * 2, F_int=base_channel),
            AttentionGate(F_g=base_channel * 4, F_l=base_channel * 4, F_int=base_channel * 2),
        ])

    def forward(self, x):
        # Encoding path
        x1 = self.feat_extract[0](x)
        res1 = self.Encoder[0](x1)

        x2 = self.feat_extract[1](res1)
        res2 = self.Encoder[1](x2)

        x3 = self.feat_extract[2](res2)
        res3 = self.Encoder[2](x3)

        # Decoding path
        d3 = self.Decoder[0](x3)
        d3 = self.feat_extract[3](d3)

        # Apply Attention Gate on skip connection
        a2 = self.AttentionGates[1](d3, res2)
        d3 = torch.cat([d3, a2], dim=1)
        d3 = self.Convs[0](d3)

        d2 = self.Decoder[1](d3)
        d2 = self.feat_extract[4](d2)

        # Apply Attention Gate on skip connection
        a1 = self.AttentionGates[0](d2, res1)
        d2 = torch.cat([d2, a1], dim=1)
        d2 = self.Convs[1](d2)

        d1 = self.Decoder[2](d2)
        d1 = self.feat_extract[5](d1)

        return d1 + x

if __name__ == "__main__":
    model = UNet(base_channel=21, num_res=6)
    count = 0
    for name,module in model.named_modules():
        print(count,'-------------',name)
        count +=1
    from functools import partial

    input = torch.randn(1, 3, 64, 64)
    output = model(input)
    print('-'*50)
    print("Output shape:",output.shape)
    print('#generator parameters:', sum(param.numel() for param in model.parameters()))
