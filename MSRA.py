import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmengine.model import BaseModule
from Models.SRA import SRA

# 定义深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 定义ConvLSTMCell带有残差和归一化
class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size,
        bias=True,
        norm='group',
        num_groups=32
    ):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.bias = bias
        self.norm_type = norm
        self.num_groups = num_groups

        self.conv = DepthwiseSeparableConv(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        if self.norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=self.num_groups, num_channels=4 * self.hidden_channels
            )
        elif self.norm_type == 'layer':
            self.norm = nn.LayerNorm(4 * self.hidden_channels)  # 修正后的LayerNorm

        self.residual_conv = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=1,
            padding=0,
            bias=False
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.conv.depthwise, self.conv.pointwise, self.residual_conv]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input, h_prev, c_prev):
        combined = torch.cat([input, h_prev], dim=1)  # 沿着通道轴拼接
        conv_output = self.conv(combined)
        if self.norm_type:
            conv_output = self.norm(conv_output)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            conv_output, self.hidden_channels, dim=1
        )
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        o = torch.sigmoid(cc_o)  # 输出门
        g = torch.tanh(cc_g)     # 细胞状态更新

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        # 残差连接
        residual = self.residual_conv(h_prev)
        h_next = h_next + residual

        return h_next, c_next

# 定义ConvLSTM
class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size,
        num_layers=1,
        bias=True
    ):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = (
            hidden_channels if isinstance(hidden_channels, list) else [hidden_channels]
        )
        self.kernel_size = kernel_size
        self.bias = bias

        # 创建多个 ConvLSTMCell 层
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            curr_input_channels = (
                input_channels if i == 0 else self.hidden_channels[i - 1]
            )
            self.layers.append(
                ConvLSTMCell(
                    input_channels=curr_input_channels,
                    hidden_channels=self.hidden_channels[i],
                    kernel_size=self.kernel_size,
                    bias=self.bias
                )
            )

    def forward(self, x, hidden_state=None):
        # x 的形状: [batch_size, channels, height, width]
        batch_size, _, height, width = x.size()

        if hidden_state is None:
            h = []
            c = []
            for i in range(self.num_layers):
                h.append(
                    torch.zeros(
                        batch_size,
                        self.hidden_channels[i],
                        height,
                        width,
                        device=x.device
                    )
                )
                c.append(
                    torch.zeros(
                        batch_size,
                        self.hidden_channels[i],
                        height,
                        width,
                        device=x.device
                    )
                )
            hidden_state = (h, c)

        h, c = hidden_state

        new_h = []
        new_c = []
        for i, layer in enumerate(self.layers):
            h_i, c_i = layer(x, h[i], c[i])
            new_h.append(h_i)
            new_c.append(c_i)
            x = h_i  # 下一层的输入是当前层的隐藏状态

        return new_h, new_c

    @staticmethod
    def init_hidden(batch_size, hidden_channels, height, width, device='cuda'):
        h = []
        c = []
        for hidden_dim in hidden_channels:
            h.append(torch.zeros(batch_size, hidden_dim, height, width, device=device))
            c.append(torch.zeros(batch_size, hidden_dim, height, width, device=device))
        return (h, c)

class RAMOptimized(nn.Module):
    def __init__(
        self,
        input_channels=64,
        hidden_channels=[64],
        kernel_size=3,
        scale_factor=1
    ):
        super(RAMOptimized, self).__init__()
        self.scale_factor = scale_factor

        # 四个尺度的ConvLSTM和SCSA模块
        self.conv_lstm_1_16 = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=len(hidden_channels),
            bias=True
        )
        self.conv_lstm_1_8 = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=len(hidden_channels),
            bias=True
        )
        self.conv_lstm_1_4 = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=len(hidden_channels),
            bias=True
        )
        self.conv_lstm_1_1 = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=len(hidden_channels),
            bias=True
        )

        # 四个尺度的SCSA模块
        self.attn_module_1_16 = SCSA(
            dim=hidden_channels[-1],
            head_num=8,
            window_size=7,
            group_kernel_sizes=[3, 5, 7, 9],
            qkv_bias=True,
            fuse_bn=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            down_sample_mode='avg_pool',
            attn_drop_ratio=0.1,
            gate_layer='sigmoid'
        )
        self.attn_module_1_8 = SCSA(
            dim=hidden_channels[-1],
            head_num=8,
            window_size=7,
            group_kernel_sizes=[3, 5, 7, 9],
            qkv_bias=True,
            fuse_bn=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            down_sample_mode='avg_pool',
            attn_drop_ratio=0.1,
            gate_layer='sigmoid'
        )
        self.attn_module_1_4 = SCSA(
            dim=hidden_channels[-1],
            head_num=8,
            window_size=7,
            group_kernel_sizes=[3, 5, 7, 9],
            qkv_bias=True,
            fuse_bn=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            down_sample_mode='avg_pool',
            attn_drop_ratio=0.1,
            gate_layer='sigmoid'
        )
        self.attn_module_1_1 = SCSA(
            dim=hidden_channels[-1],
            head_num=8,
            window_size=7,
            group_kernel_sizes=[3, 5, 7, 9],
            qkv_bias=True,
            fuse_bn=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            down_sample_mode='avg_pool',
            attn_drop_ratio=0.1,
            gate_layer='sigmoid'
        )

        # 四个尺度的预测卷积
        self.conv_pred_1_16 = nn.Conv2d(
            hidden_channels[-1],
            input_channels,
            kernel_size=1,
            padding=0
        )
        self.conv_pred_1_8 = nn.Conv2d(
            hidden_channels[-1],
            input_channels,
            kernel_size=1,
            padding=0
        )
        self.conv_pred_1_4 = nn.Conv2d(
            hidden_channels[-1],
            input_channels,
            kernel_size=1,
            padding=0
        )
        self.conv_pred_1_1 = nn.Conv2d(
            hidden_channels[-1],
            input_channels,
            kernel_size=1,
            padding=0
        )

        # 融合卷积层
        self.fusion_conv = nn.Conv2d(
            input_channels * 4,  # 四个尺度的输入通道数总和
            input_channels,      # 输出通道数
            kernel_size=1,
            padding=0,
            bias=False
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                if hasattr(m, 'weight'):
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)

    def forward(self, saliency_maps):
        optimized_maps = {}

        saliency_map_1_16 = saliency_maps['1_16']
        h_1_16, c_1_16 = self.conv_lstm_1_16(saliency_map_1_16)
        lstm_feature_1_16 = h_1_16[-1]
        attn_feature_1_16 = self.attn_module_1_16(lstm_feature_1_16)
        optimized_1_16 = self.conv_pred_1_16(attn_feature_1_16)
        optimized_maps['1_16'] = optimized_1_16

        # 处理 1/8 尺度
        saliency_map_1_8 = saliency_maps['1_8']
        h_1_8, c_1_8 = self.conv_lstm_1_8(saliency_map_1_8)
        lstm_feature_1_8 = h_1_8[-1]
        attn_feature_1_8 = self.attn_module_1_8(lstm_feature_1_8)
        optimized_1_8 = self.conv_pred_1_8(attn_feature_1_8)
        optimized_maps['1_8'] = optimized_1_8

        # 处理 1/4 尺度
        saliency_map_1_4 = saliency_maps['1_4']
        h_1_4, c_1_4 = self.conv_lstm_1_4(saliency_map_1_4)
        lstm_feature_1_4 = h_1_4[-1]
        attn_feature_1_4 = self.attn_module_1_4(lstm_feature_1_4)
        optimized_1_4 = self.conv_pred_1_4(attn_feature_1_4)
        optimized_maps['1_4'] = optimized_1_4

        saliency_map_1_1 = saliency_maps['1_1']


        hw = int(saliency_map_1_1.size(1) ** 0.5) 
        assert hw * hw == saliency_map_1_1.size(1), \
            f"特征图尺寸 {saliency_map_1_1.size(1)}
        # 调整维度顺序和形状
        saliency_map_1_1 = saliency_map_1_1.permute(0, 2, 1)  # [B, C, H*W]
        saliency_map_1_1 = saliency_map_1_1.view(
            saliency_map_1_1.size(0),
            saliency_map_1_1.size(1),  # C
            hw,  # H
            hw  # W
        )  # 最终形状 [B, C, H, W]

        # 处理修正后的 1/1 尺度
        h_1_1, c_1_1 = self.conv_lstm_1_1(saliency_map_1_1)
        lstm_feature_1_1 = h_1_1[-1]
        attn_feature_1_1 = self.attn_module_1_1(lstm_feature_1_1)
        optimized_1_1 = self.conv_pred_1_1(attn_feature_1_1)
        optimized_maps['1_1'] = optimized_1_1


        target_height = saliency_map_1_1.size(2)  # H
        target_width = saliency_map_1_1.size(3)  # W
        optimized_1_16_up = F.interpolate(
            optimized_maps['1_16'],
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=True
        )
        optimized_1_8_up = F.interpolate(
            optimized_maps['1_8'],
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=True
        )
        optimized_1_4_up = F.interpolate(
            optimized_maps['1_4'],
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=True
        )
        optimized_1_1 = optimized_maps['1_1'] 

        concatenated = torch.cat([
            optimized_1_16_up,
            optimized_1_8_up,
            optimized_1_4_up,
            optimized_1_1
        ], dim=1)  # [B, C*4, H, W]

        fused_optimized = self.fusion_conv(concatenated)  # [B, C, H, W]
        return fused_optimized
