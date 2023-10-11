# coding: utf8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

from gcn_lib import act_layer, batched_index_select
from modules.pyramid_vig import pvig_ti_224_gelu, pvig_s_224_gelu, pvig_m_224_gelu, pvig_b_224_gelu
from tools.utils import setup_seed


class F2P(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
            act_layer(act)
        )
        self.fc2 = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
            nn.Softmax(dim=1)
        )
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Distribution(nn.Module):

    def __init__(self):
        super(Distribution, self).__init__()

    def forward(self, x, x_p, edge_index):
        # print(x.shape)
        # print(y==None)
        n_B, n_C, n_N, _ = x.shape
        c1 = 1e-6
        c2 = 1e-6
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        xp_i = batched_index_select(x_p, edge_index[1])
        xp_j = batched_index_select(x_p, edge_index[0])

        x_i_mean = torch.mean(x_i, dim=1, keepdim=True)
        x_j_mean = torch.mean(x_j, dim=1, keepdim=True)

        i_var = ((x_i - x_i_mean) ** 2).mean(dim=1, keepdim=True)
        j_var = ((x_j - x_j_mean) ** 2).mean(dim=1, keepdim=True)

        ij_mul = torch.mul(x_i, x_j)
        ij_covs = ij_mul.mean(dim=1, keepdim=True) - x_i_mean * x_j_mean

        S1 = (2 * x_i_mean * x_j_mean + c1) / (x_i_mean ** 2 + x_j_mean ** 2 + c1)
        S2 = (2 * ij_covs + c2) / (i_var + j_var + c2)
        sff = 1 - S1 * S2

        xps_ij = torch.abs(xp_i - xp_j)

        sum_xx = torch.sum(xps_ij * sff, dim=-1)
        sum_x = torch.sum(xp_i, dim=-1) + torch.sum(xp_j, dim=-1)

        Ex = sum_x + sum_xx
        # print(Ex.shape)
        return Ex.unsqueeze(-1)


class GRIDS(torch.nn.Module):
    def __init__(self, device=torch.device('cpu'), type='ti', drop_path_rate=0.0):
        super(GRIDS, self).__init__()

        self.device = device

        if type == 'ti':
            self.feature_extractor = pvig_ti_224_gelu(drop_path_rate=drop_path_rate)
        elif type == 's':
            self.feature_extractor = pvig_s_224_gelu(drop_path_rate=drop_path_rate)
        elif type == 'm':
            self.feature_extractor = pvig_m_224_gelu(drop_path_rate=drop_path_rate)
        elif type == 'b':
            self.feature_extractor = pvig_b_224_gelu(drop_path_rate=drop_path_rate)
        else:
            raise Exception('wrong model type')

        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        blocks = self.feature_extractor.blocks
        channels = self.feature_extractor.channels
        self.down_nums = [blocks[0], sum(blocks[0:2]) + 1, sum(blocks[0:3]) + 2]
        self.graph_nums = [blocks[0] - 1, sum(blocks[0:2]), sum(blocks[0:3]) + 1, sum(blocks) + 2]  # 三个下采样

        self._fp = F2P(channels[0], act='gelu')
        self.fp0 = F2P(channels[0], act='gelu')
        self.fp1 = F2P(channels[1], act='gelu')
        self.fp2 = F2P(channels[2], act='gelu')
        self.fp3 = F2P(channels[3], act='gelu')

        self.fps = [self.fp0, self.fp1, self.fp2, self.fp3]

        self.distribution = Distribution()
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, 5)))
        self.alpha.data.normal_(1, 0.1)

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def JSD_all(self, A, B):
        assert A.shape == B.shape

        n_B, n_C, n_N, _ = A.shape
        if _ != 1:
            A = A.reshape(n_B, n_C, -1, 1).contiguous()
            B = B.reshape(n_B, n_C, -1, 1).contiguous()

        A = torch.squeeze(A.transpose(2, 1), dim=3)
        B = torch.squeeze(B.transpose(2, 1), dim=3)

        M = (A + B) / 2

        jsd = 0.5 * torch.sum(A * torch.log(A / M), dim=2) + 0.5 * torch.sum(B * torch.log(B / M), dim=2)
        jsd = torch.mean(jsd, dim=1)
        return jsd

    def Hellinger_distance_all(self, A, B):
        assert A.shape == B.shape

        n_B, n_C, n_N, _ = A.shape
        if _ != 1:
            A = A.reshape(n_B, n_C, -1, 1).contiguous()
            B = B.reshape(n_B, n_C, -1, 1).contiguous()

        A = torch.squeeze(A.transpose(2, 1), dim=3)  # n_B, n_N, n_C
        B = torch.squeeze(B.transpose(2, 1), dim=3)

        hds = 1 / math.sqrt(2) * torch.norm(torch.sqrt(A) - torch.sqrt(B), dim=2)
        hds = torch.where(torch.isnan(hds), torch.full_like(hds, 0), hds)
        hds = torch.mean(hds, dim=1)

        return hds

    def forward(self, ref_img, dst_img):

        distances = []

        x = self.feature_extractor.stem(ref_img) + self.feature_extractor.pos_embed
        y = self.feature_extractor.stem(dst_img) + self.feature_extractor.pos_embed

        B, C, H, W = x.shape
        _x = x.reshape(B, C, -1, 1).contiguous()
        _y = y.reshape(B, C, -1, 1).contiguous()
        _x_p = self._fp(_x)
        _y_p = self._fp(_y)

        # dis = self.JSD_all(_x_p, _y_p)
        dis = self.Hellinger_distance_all(_x_p, _y_p)
        distances.append(dis)

        g_i = 0
        for i in range(len(self.feature_extractor.backbone)):
            # print(i, self.feature_extractor.backbone[i])
            if i in self.down_nums:
                x = self.feature_extractor.backbone[i](x)
                y = self.feature_extractor.backbone[i](y)
            else:
                grapher = self.feature_extractor.backbone[i][0]

                B, C, H, W = x.shape

                x = grapher(x, calculate_edge_index=True)
                edge_index = grapher.graph_conv.edge_index
                y = grapher(y, calculate_edge_index=False)

                if i == self.graph_nums[g_i]:
                    _x = x.reshape(B, C, -1, 1).contiguous()
                    _y = y.reshape(B, C, -1, 1).contiguous()

                    _x_p = self.fps[g_i](_x)
                    _y_p = self.fps[g_i](_y)

                    Px = self.distribution(_x, _x_p, edge_index)
                    Py = self.distribution(_y, _y_p, edge_index)
                    dis = self.Hellinger_distance_all(Px, Py)
                    # dis = self.JSD_all(Px, Py)

                    # dis = self.Hellinger_distance_all(_x_p, _y_p)
                    # dis = self.JSD_all(_x_p, _y_p)

                    distances.append(dis)

                    g_i = g_i + 1

                x = self.feature_extractor.backbone[i][1](x)
                y = self.feature_extractor.backbone[i][1](y)

        distances = torch.stack(distances).transpose(1, 0).to(self.device)
        weighted_distances = self.alpha * distances
        mean_distances = torch.mean(weighted_distances, dim=1)
        score = 1 - mean_distances

        return score
