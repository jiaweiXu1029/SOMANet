import torch.nn as nn
import torch
from token_performer import Token_performer
from Transformer import saliency_token_inference, contour_token_inference, token_TransformerEncoder
from ram import RAMOptimized


class token_trans(nn.Module):
    def __init__(self, in_dim=64, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_trans, self).__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.saliency_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1)
        self.contour_token_pre = contour_token_inference(dim=embed_dim, num_heads=1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.norm1_c = nn.LayerNorm(embed_dim)
        self.mlp1_c = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.norm2_c = nn.LayerNorm(embed_dim)
        self.mlp2_c = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )


    def forward(self, fea, saliency_tokens, contour_tokens, sal_PE, con_PE, back_PE, depth_fea, zoom, mask):
        B, _, _ = fea.shape
        # fea [B, H*W, 64]
        # project to 384 dim
        fea = self.mlp(self.norm(fea))
        # fea [B, H*W, 384]

        fea,saliency_tokens,contour_tokens,fea_tmp,fea_s = self.encoderlayer(fea, saliency_tokens, contour_tokens, sal_PE, con_PE, back_PE, depth_fea, zoom, mask)

        # reproject back to 64 dim
        saliency_tokens_tmp = self.mlp1(self.norm1(saliency_tokens))
        contour_tokens_tmp = self.mlp1_c(self.norm1_c(contour_tokens))
        
        saliency_fea = self.saliency_token_pre(fea_s)
        # saliency_fea [B, H*W, 384]
        contour_fea = self.contour_token_pre(fea_s)
        # contour_fea [B, H*W, 384]

        # reproject back to 64 dim
        saliency_fea = self.mlp2(self.norm2(saliency_fea))
        contour_fea = self.mlp2_c(self.norm2_c(contour_fea))

        return fea, saliency_tokens, contour_tokens, fea_tmp, saliency_tokens_tmp, contour_tokens_tmp, saliency_fea, contour_fea


class decoder_module(nn.Module):
    def __init__(self, dim=384, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True):
        super(decoder_module, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio,  img_size // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim*2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
           

            # project input feature to 64 dim
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
                nn.Linear(dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )

    def forward(self, dec_fea, enc_fea=None):
    
        B,_,C = dec_fea.shape
        if C == 384:
            dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        # [B, HW, C]

        if self.fuse:
            # [2] fuse encoder fea and decoder fea
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
        return dec_fea


class Decoder(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, depth=2, img_size=224):

        super(Decoder, self).__init__()

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm2_c = nn.LayerNorm(embed_dim)
        self.mlp2_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm_c = nn.LayerNorm(embed_dim)
        self.mlp_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.img_size = img_size
        self.token_dim = token_dim
        # token upsampling and multi-level token fusion
        self.decoder1 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder2 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder3 = decoder_module(dim=embed_dim, token_dim=token_dim,  img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)
        self.decoder3_s = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)
        self.decoder3_c = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)
        

        # token based multi-task predictions
        self.token_pre_1_8 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)
        self.token_pre_1_4 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)

        self.pre_1_16 = nn.Linear(token_dim, 1)
        self.pre_1_8 = nn.Linear(token_dim, 1)
        self.pre_1_4 = nn.Linear(token_dim, 1)
        self.pre_1_1 = nn.Linear(token_dim, 1)
        # predict contour maps
        self.pre_1_16_c = nn.Linear(token_dim, 1)
        self.pre_1_8_c = nn.Linear(token_dim, 1)
        self.pre_1_4_c = nn.Linear(token_dim, 1)
        self.pre_1_1_c = nn.Linear(token_dim, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True)
        
        self.zoom_8 = nn.Parameter(torch.ones(1)*0.5)
        self.zoom_4 = nn.Parameter(torch.ones(1)*0.5)
        
        self.enc_dim_8 = nn.Linear(embed_dim//2, token_dim)
        self.enc_dim_4 = nn.Linear(embed_dim//4, token_dim)
        # 实例化 RAMOptimized 模块，仅用于显著性特征图优化
        self.ram_saliency = RAMOptimized(
            input_channels=64,  # 与 fea_* 的通道数一致
            hidden_channels=[64],  # 隐藏通道数
            kernel_size=3,  # 卷积核大小
            scale_factor=1  # 保持空间尺寸不变
        )
        for m in self.modules():
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                if hasattr(m, 'weight'):
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)


    def forward(self, fea_1_16, saliency_tokens, contour_tokens, fea_16, saliency_tokens_16, contour_tokens_16, saliency_fea_1_16, contour_fea_1_16,rgb_fea_1_8, rgb_fea_1_4, depth_28, depth_56, sal_PE, con_PE, back_PE):
        # saliency_fea_1_16 [B, 14*14, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # token_fea_1_16  [B, 1 + 14*14 + 1, 384] (contain saliency token and contour token)

        # saliency_tokens [B, 1, 384]
        # contour_tokens [B, 1, 384]

        # rgb_fea_1_8 [B, 28*28, 64]
        # rgb_fea_1_4 [B, 28*28, 64]

        B, _, _, = fea_1_16.size()
        dim_d = torch.sqrt(torch.tensor(self.token_dim).to(torch.double))
        
        mask_1_16 = (fea_16 @ (saliency_tokens_16.permute(0,2,1)))/dim_d
        mask_1_16 = mask_1_16.reshape(B, 1, self.img_size // 16, self.img_size // 16)
    
        con_1_16 = (fea_16 @ contour_tokens_16.permute(0,2,1))/dim_d
        con_1_16 = con_1_16.reshape(B, 1, self.img_size // 16, self.img_size // 16)
        
        saliency_fea_1_16 = self.mlp(self.norm(saliency_fea_1_16))
        # saliency_fea_1_16 [B, 14*14, 64]
        mask_1_16_s = self.pre_1_16(saliency_fea_1_16)
        mask_1_16_s = mask_1_16_s.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        contour_fea_1_16 = self.mlp_c(self.norm_c(contour_fea_1_16))
        # contour_fea_1_16 [B, 14*14, 64]
        con_1_16_s = self.pre_1_16_c(contour_fea_1_16)
        con_1_16_s = con_1_16_s.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        # 1/16 -> 1/8
        # reverse T2T and fuse low-level feature
        rgb_fea_1_8 = self.enc_dim_8(rgb_fea_1_8)
        fea_1_8 = self.decoder1(fea_1_16, rgb_fea_1_8)
        mask_8 = self.upsample(mask_1_16.sigmoid()).reshape(B,1,-1)
        # token prediction
        fea_1_8, saliency_tokens, contour_tokens, fea_8, saliency_tokens_8, contour_tokens_8,  saliency_fea_1_8, contour_fea_1_8  = self.token_pre_1_8(fea_1_8, saliency_tokens, contour_tokens, sal_PE, con_PE, back_PE, depth_28, self.zoom_8, mask_8)
        # predict saliency maps and contour maps
        mask_1_8 = (fea_8 @ (saliency_tokens_8.permute(0,2,1)))/dim_d
        mask_1_8 = mask_1_8.reshape(B, 1, self.img_size // 8, self.img_size // 8)
    
        con_1_8 = (fea_8 @ contour_tokens_8.permute(0,2,1))/dim_d
        con_1_8 = con_1_8.reshape(B, 1, self.img_size // 8, self.img_size // 8)
        
        mask_1_8_s = self.pre_1_8(saliency_fea_1_8)
        mask_1_8_s = mask_1_8_s.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)

        con_1_8_s = self.pre_1_8_c(contour_fea_1_8)
        con_1_8_s = con_1_8_s.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)

        # 1/8 -> 1/4
        rgb_fea_1_4 = self.enc_dim_4(rgb_fea_1_4)
        fea_1_4 = self.decoder2(fea_1_8, rgb_fea_1_4)
        mask_4 = self.upsample(mask_1_8.sigmoid()).reshape(B,1,-1)
        fea_1_4, saliency_tokens, contour_tokens, fea_4, saliency_tokens_4, contour_tokens_4,saliency_fea_1_4, contour_fea_1_4 = self.token_pre_1_4(fea_1_4, saliency_tokens, contour_tokens, sal_PE, con_PE,  back_PE, depth_56, self.zoom_4, mask_4)
        # predict saliency maps and contour maps
        mask_1_4 = (fea_4 @ saliency_tokens_4.permute(0,2,1))/dim_d
        mask_1_4 = mask_1_4.reshape(B, 1, self.img_size // 4, self.img_size // 4)

        con_1_4 = (fea_4 @ contour_tokens_4.permute(0,2,1))/dim_d
        con_1_4 = con_1_4.reshape(B, 1, self.img_size // 4, self.img_size // 4)
        
         
        mask_1_4_s = self.pre_1_4(saliency_fea_1_4)
        mask_1_4_s = mask_1_4_s.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)

        con_1_4_s = self.pre_1_4_c(contour_fea_1_4)
        con_1_4_s = con_1_4_s.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)
        
        # 1/4 -> 1
        fea_1_1 = self.decoder3(fea_1_4)
        saliency_fea_1_1 = self.decoder3_s(saliency_fea_1_4)
        contour_fea_1_1 = self.decoder3_c(contour_fea_1_4)
        saliency_tokens_1 = self.mlp2(self.norm2(saliency_tokens))
        contour_tokens_1 = self.mlp2_c(self.norm2_c(contour_tokens))

        # 收集所有四个尺度的显著性特征图
        saliency_maps = {
            '1_16': saliency_fea_1_16.transpose(1, 2).view(B, 64, self.img_size // 16, self.img_size // 16),
            '1_8': saliency_fea_1_8.transpose(1, 2).view(B, 64, self.img_size // 8, self.img_size // 8),
            '1_4': saliency_fea_1_4.transpose(1, 2).view(B, 64, self.img_size // 4, self.img_size // 4),
            '1_1': saliency_fea_1_1
        }

        # 优化显著性特征图
        optimized_fused_map = self.ram_saliency(saliency_maps)  # [B, 64, H, W]


        # predict saliency maps and contour maps
        mask_1_1 = (fea_1_1 @ saliency_tokens_1.permute(0,2,1))/dim_d
        mask_1_1 = mask_1_1.reshape(B, 1, self.img_size, self.img_size )

        con_1_1 = (fea_1_1 @ contour_tokens_1.permute(0,2,1))/dim_d
        con_1_1 = con_1_1.reshape(B, 1, self.img_size, self.img_size )
        

        mask_1_1_s = self.pre_1_1(optimized_fused_map.view(B, 64, -1).transpose(1, 2))  # [B, HW, 1]
        mask_1_1_s = mask_1_1.transpose(1, 2).reshape(B, 1, self.img_size, self.img_size)

        con_1_1_s = self.pre_1_1_c(contour_fea_1_1)
        con_1_1_s = con_1_1_s.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)
        #[mask_1_32,mask_1_16, mask_1_8, mask_1_4, mask_1_1], [contour_1_32,contour_1_16, contour_1_8, contour_1_4, contour_1_1]
        return [mask_1_16, mask_1_8, mask_1_4, mask_1_1], [con_1_16, con_1_8, con_1_4, con_1_1],[mask_1_16_s, mask_1_8_s, mask_1_4_s, mask_1_1_s], [con_1_16_s, con_1_8_s, con_1_4_s, con_1_1_s]

