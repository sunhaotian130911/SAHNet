import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

from .misc import NestedTensor
from .DeformTrans3D.DeformableTrans3D import DeformableTransformer3D
from .DeformTrans3D.position_encoding_3D import PositionEmbeddingSine_3D

def conv3x3_2D(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=8, bias=False)
def conv3x3_3D(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=10, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool3d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool3d( x, 2, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        # return x * scale
        return scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
class GroupBlock(nn.Module):
    expansion = 1
    dim = 2
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(GroupBlock, self).__init__()
        self.conv1 = conv3x3_2D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.L_relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.conv2 = conv3x3_2D(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.L_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.L_relu(out)
        return out

class NormalBlock(nn.Module):
    expansion = 1
    dim = 3
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(NormalBlock, self).__init__()
        self.conv1 = conv3x3_3D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.L_relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.conv2 = conv3x3_3D(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.L_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.L_relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block1, block2, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(8, self.inplanes, kernel_size=7, stride=1, padding=3, groups=8,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.L_relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.layer1_1 = self._make_layer(block1, 128, layers[0], stride=2)
        self.layer1_2 = self._make_layer(block1, 256, layers[1], stride=2)
        self.layer1_3 = self._make_layer(block1, 512, layers[2], stride=2) # 1536
        # 2D to 3D
        self.transfrom_layer = self._transfrom #channal number = layer2[layer number]/8, 128/8=16
        # be used to stay the same channel nums of different scale feature maps
        self.conv1_4 = nn.Conv3d(16, 120, kernel_size=1, stride=1)
        self.conv2_4 = nn.Conv3d(32, 120, kernel_size=1, stride=1)
        self.conv3_4 = nn.Conv3d(64, 120, kernel_size=1, stride=1)

        self.att_mask2 = CBAM(gate_channels=120,reduction_ratio = 10, pool_types = ['avg', 'max'], no_spatial = False)
        self.att_mask3 = CBAM(gate_channels=120,reduction_ratio = 10, pool_types = ['avg', 'max'], no_spatial = False)
        self.att_mask4 = CBAM(gate_channels=120,reduction_ratio = 10, pool_types = ['avg', 'max'], no_spatial = False)

        self.UpSample_mask3 = torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.UpSample_mask4 = torch.nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)

        # Deform Trans
        self.Dtransformer = DeformableTransformer3D(d_model=120, nhead=8, num_encoder_layers=6, dim_feedforward=256,
                                              dropout=0.1, activation='relu', num_feature_levels=3, enc_n_points=4)
        self.DPOS = PositionEmbeddingSine_3D(num_pos_feats=[40, 40, 40])

        self.layer2_1 = self._make_layer(block2, 16, layers[3], stride=2)
        self.layer2_2 = self._make_layer(block2, 32, layers[4], stride=2)
        self.layer2_3 = self._make_layer(block2, 64, layers[3], stride=2)

        self.GlobalMaxpool3D = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Linear(360 * block2.expansion, 512 * block2.expansion)
        self.fc2 = nn.Linear(512 * block2.expansion, num_classes)

    def _transfrom(self,x,groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups,
                   channels_per_group, height, width)
        # transpose
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        # x = x.view(batchsize, -1, height, width)
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if block.dim ==2:
                downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes * block.expansion),
                )
            elif block.dim == 3:
                if self.inplanes*2 != planes*block.expansion:
                    self.inplanes = self.inplanes//32
                else:
                    self.inplanes = self.inplanes*2
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=(stride,stride,stride), bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.L_relu(x)

        x_1 = self.layer1_1(x)
        x_2 = self.layer1_2(x_1)
        x_3 = self.layer1_3(x_2)

        x_1 = self.transfrom_layer(x_1, groups=8)
        x_1 = self.layer2_1(x_1)
        x_1 = self.conv1_4(x_1)

        x_2 = self.transfrom_layer(x_2, groups=8)
        x_2 = self.layer2_2(x_2)
        x_2 = self.conv2_4(x_2)

        x_3 = self.transfrom_layer(x_3, groups=8)
        x_3 = self.layer2_3(x_3)
        x_3 = self.conv3_4(x_3)

        ''' Multi Scale Deformable Transformer '''
        x_allscale = [x_1, x_2, x_3]
        att_mask = []
        masks = []
        pos_Ds = []
        shape_tmp = []
        for i, x_tmp in enumerate(x_allscale):
            b, c, d, h, w = x_tmp.shape
            shape_tmp.append([d,h,w])
            if i == 0:
                mask = self.att_mask2(x_tmp)
            if i == 1:
                mask = self.att_mask3(x_tmp)
            if i == 2:
                mask = self.att_mask4(x_tmp)
            att_mask.append(mask.squeeze(1))
            mask = torch.where(mask > 0.5, torch.ones_like(mask, dtype=torch.uint8), torch.zeros_like(mask, dtype=torch.uint8))
            masks.append(mask.squeeze(1))
            pos_D = self.DPOS(x_tmp)
            pos_Ds.append(pos_D)

        att_mask[1] = self.UpSample_mask3(att_mask[1])
        att_mask[2] = self.UpSample_mask4(att_mask[2])

        x = self.Dtransformer(x_allscale, masks, pos_Ds)

        # 256 input
        x_1_A, x_2_A, x_3_A = x.split((shape_tmp[0][0]*shape_tmp[0][1]*shape_tmp[0][2],
                                       shape_tmp[1][0]*shape_tmp[1][1]*shape_tmp[1][2],
                                       shape_tmp[2][0]*shape_tmp[2][1]*shape_tmp[2][2]), dim=1) #[4,64,64],[4,32,32],[4,16,16]

        x_1 = x_1_A.permute(0,2,1).view(b, c, shape_tmp[0][0], shape_tmp[0][1], shape_tmp[0][2]) + x_1
        x_2 = x_2_A.permute(0,2,1).view(b, c, shape_tmp[1][0], shape_tmp[1][1], shape_tmp[1][2]) + x_2
        x_3 = x_3_A.permute(0,2,1).view(b, c, shape_tmp[2][0], shape_tmp[2][1], shape_tmp[2][2]) + x_3

        x_22 = self.GlobalMaxpool3D(x_1)
        x_33 = self.GlobalMaxpool3D(x_2)
        x_44 = self.GlobalMaxpool3D(x_3)

        x = torch.cat((x_22, x_33, x_44),dim=1)

        x = x.view(x.size(0), -1) #shape 4 * 256, batch * channel
        x = self.fc1(x)
        x = self.fc2(x)
        return x, att_mask

def SAHNet_ResNet18(**kwargs):
    model = ResNet(GroupBlock, NormalBlock, [2, 2, 2, 2], **kwargs)
    return model

