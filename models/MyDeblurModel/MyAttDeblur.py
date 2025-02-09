import torch.nn as nn
import torch
import torch.nn.functional as F
from models.MyDeblurModel.block import UpBlock, Init_Conv, Init_Conv_V2, Conv1x1, ResBlock, Init_Conv_V3, NAFInitBlock, SP_block
import timm
# from models.Uformer.model import LeWinTransformerBlock
from models.MLWNet.MLWNet_arch import WaveletBlock
from models.NAFNet.NAFNet_arch import NAFBlock
from einops import rearrange, repeat
import Deblur_Utils.down_core as down_core


class MyADeblurModel(nn.Module):
    def __init__(self, in_channel=3, width=16, num_main=20, num_encoder=1):
        super(MyADeblurModel, self).__init__()
        self.width = width
        self.num_layers = num_main + (6 * num_encoder)

        self.began = nn.Conv2d(in_channel, width, 3, 1, 1, bias=True)
        self.encoders = nn.ModuleList([
            nn.Sequential(*[Init_Conv(width) for _ in range(num_encoder)]),
            nn.Sequential(*[Init_Conv(width * 2) for _ in range(num_encoder)]),
            nn.Sequential(*[Init_Conv(width * 4) for _ in range(num_encoder)]),
            Init_Conv(width * 8),
        ])
        self.downs = nn.ModuleList([
            nn.Conv2d(width, width * 2, 2, 2, 0),
            nn.Conv2d(width * 2, width * 4, 2, 2, 0),
            nn.Conv2d(width * 4, width * 8, 2, 2, 0)
        ])
        self.main = nn.Sequential(*[Init_Conv(width * 8) for _ in range(num_main)])
        self.ups = nn.ModuleList([
            UpBlock(width * 8),
            UpBlock(width * 4),
            UpBlock(width * 2)
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(*[Init_Conv(width * 4) for _ in range(num_encoder)]),
            nn.Sequential(*[Init_Conv(width * 2) for _ in range(num_encoder)]),
            nn.Sequential(*[Init_Conv(width) for _ in range(num_encoder)]),
        ])
        self.end = nn.Conv2d(width, in_channel, 3, 1, 1, bias=True)

    def forward(self, x):
        out1 = self.began(x)  # wd
        en1 = self.encoders[0](out1)

        en2 = self.downs[0](en1)  # 2w
        en2 = self.encoders[1](en2)

        en3 = self.downs[1](en2)  # 4w
        en3 = self.encoders[2](en3)

        en4 = self.downs[2](en3)  # 8w
        en4 = self.encoders[3](en4)
        en4 = self.main(en4)

        de1 = self.ups[0](en4)  # 4w
        de1 = self.decoders[0](de1 + en3)

        de2 = self.ups[1](de1)  # 2w
        de2 = self.decoders[1](de2 + en2)

        de3 = self.ups[2](de2)  # w
        de3 = self.decoders[2](de3 + en1)

        out = self.end(de3)

        return out + x


class MyADeblurModel_V2(nn.Module):
    def __init__(self, in_channel=3, width=16, num_main=20, up_scale=8, num_encoder=1):
        super(MyADeblurModel_V2, self).__init__()
        self.width = width
        self.num_layers = num_main + (num_encoder * 3)
        self.began = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1),
            nn.Sequential(*[Init_Conv_V2(width, num_heads=16) for _ in range(num_encoder)]),
            # nn.Hardtanh(min_val=-0.843, max_val=0.906),

            nn.Conv2d(width, width * 2, 2, 2, 0),
            nn.Sequential(*[Init_Conv_V2(width * 2, num_heads=8) for _ in range(num_encoder)]),
        )
        if up_scale == 4:
            self.began.append(nn.Conv2d(width * 2, width * 4, 2, 2, 0))
            self.began.append(nn.Sequential(*[Init_Conv_V2(width * 4, num_heads=4) for _ in range(num_encoder)]))

        if up_scale == 8:
            self.began.append(nn.Conv2d(width * 2, width * 4, 2, 2, 0))
            self.began.append(nn.Sequential(*[Init_Conv_V2(width * 4, num_heads=4) for _ in range(num_encoder)]))

            self.began.append(nn.Conv2d(width * 4, width * up_scale, 2, 2, 0))
            self.began.append(nn.Sequential(*[Init_Conv_V2(width * up_scale, num_heads=2) for _ in range(num_encoder)]))

        if up_scale == 16:
            self.began.append(nn.Conv2d(width * 2, width * 4, 2, 2, 0))
            self.began.append(nn.Sequential(*[Init_Conv_V2(width * 4, num_heads=4) for _ in range(num_encoder)]))

            self.began.append(nn.Conv2d(width * 4, width * 8, 2, 2, 0))
            self.began.append(nn.Sequential(*[Init_Conv_V2(width * 8, num_heads=2) for _ in range(num_encoder)]), )
            self.began.append(nn.Conv2d(width * 8, width * up_scale, 2, 2, 0))

        self.main = nn.Sequential(
            # nn.PixelUnshuffle(2),
            *[Init_Conv_V2(width * up_scale, num_heads=4) for _ in range(num_main)],
            # nn.PixelShuffle(2)
        )
        self.end = nn.Sequential(
            nn.Conv2d(width * up_scale, 3 * (up_scale ** 2), 3, 1, 1),
            # nn.Hardtanh(min_val=-0.843, max_val=0.906),
            nn.PixelShuffle(up_scale),
        )
        # self.ca = ChannelAtt(width * up_scale)
        # self.initialize()

    def forward(self, inp):
        x1 = self.began(inp)
        x = self.main(x1)
        x = self.end(x + x1)
        # x = self.end(x1)
        return x + inp

    def expend_deep(self, block_nums):
        block = self.main[-1]
        for _ in range(block_nums):
            self.main.append(block)
        self.num_layers += block_nums


class MyADeblurModel_V3(nn.Module):
    def __init__(self, in_channel=3, width=16, num_main=20, up_scale=8, en_decoders_blocks=None):
        super(MyADeblurModel_V3, self).__init__()
        self.width = width
        if en_decoders_blocks is None:
            en_decoders_blocks = [1, 1, 1]
        self.num_layers = num_main + sum(en_decoders_blocks) + 3

        self.began = nn.Conv2d(in_channel, width, 3, 1, 1)

        self.encoder = nn.ModuleList([
            nn.Sequential(*[Init_Conv(width) for _ in range(en_decoders_blocks[0])]),
            nn.Conv2d(width, width * 2, 2, 2, 0),

            nn.Sequential(*[Init_Conv(width * 2) for _ in range(en_decoders_blocks[1])]),
            nn.Conv2d(width * 2, width * 4, 2, 2, 0),

            nn.Sequential(*[Init_Conv(width * 4) for _ in range(en_decoders_blocks[2])]),
            nn.Conv2d(width * 4, width * 8, 2, 2, 0),
        ])

        self.main = nn.Sequential(*[Init_Conv(width * 8, num_heads=4) for _ in range(num_main)], )
        self.up = nn.ModuleList([
            nn.Sequential(nn.PixelShuffle(8),
                          nn.Conv2d(width // 8, width, 1, 1, 0),),
            nn.Sequential(*[Init_Conv(width) for _ in range(en_decoders_blocks[0])]),

            nn.Sequential(nn.PixelShuffle(4),
                          nn.Conv2d(width // 4, width, 1, 1, 0),),
            nn.Sequential(*[Init_Conv(width) for _ in range(en_decoders_blocks[0])]),

            nn.Sequential(nn.PixelShuffle(2),
                          nn.Conv2d(width // 2, width, 1, 1, 0),),
            nn.Sequential(*[Init_Conv(width) for _ in range(en_decoders_blocks[0])]),
        ])

        self.end = nn.Conv2d(width, in_channel, 1, 1, 0)

    def forward(self, x):
        out1 = self.began(x)  # 1c  1wh
        out1 = self.encoder[0](out1)  # 1c  1wh

        out2 = self.encoder[1](out1)  # 2c  1/2wh
        out2 = self.encoder[2](out2)  # 2c  1/2wh

        out3 = self.encoder[3](out2)  # 4c  1/4wh
        out3 = self.encoder[4](out3)  # 4c  1/4wh

        out4 = self.encoder[5](out3)  # 8c  1/8wh
        out4 = self.main(out4)  # 8c  1/8wh

        up1 = self.up[0](out4)  # 1c  1wh
        up1 = self.up[1](up1)  # 1c  1wh

        up2 = self.up[2](out3)  # 1c  1wh
        up2 = self.up[3](up1 + up2)  # 1c  1wh

        up3 = self.up[4](out2)  # 1c  1wh
        up3 = self.up[5](up2 + up3)  # 1c  1wh

        out = self.end(up3)

        return out + x


class MyADeblurModel_V4(nn.Module):
    def __init__(self, in_channel=3, width=16, num_main=20, up_scale=8, en_decoders_blocks=None):
        super(MyADeblurModel_V4, self).__init__()
        self.width = width
        if en_decoders_blocks is None:
            en_decoders_blocks = [2, 4, 4]
        self.num_layers = num_main + sum(en_decoders_blocks) + 2

        self.began = nn.Conv2d(in_channel, width, 3, 1, 1)

        self.encoder = nn.ModuleList([
            nn.Sequential(*[Init_Conv(width) for _ in range(en_decoders_blocks[0])]),
            nn.Conv2d(width, width * 2, 2, 2, 0),

            nn.Sequential(*[Init_Conv(width * 2) for _ in range(en_decoders_blocks[1])]),
            nn.Conv2d(width * 2, width * 4, 2, 2, 0),

            nn.Sequential(*[Init_Conv(width * 4) for _ in range(en_decoders_blocks[2])]),
            nn.Conv2d(width * 4, width * 8, 2, 2, 0),
        ])

        self.main = nn.Sequential(*[Init_Conv(width * 8, num_heads=4) for _ in range(num_main)], )
        self.up = nn.ModuleList([
            nn.Sequential(nn.PixelShuffle(8)),
            nn.Sequential(nn.PixelShuffle(4)),
            nn.Sequential(nn.PixelShuffle(2)),
        ])
        self.decoder = nn.Sequential(
            nn.Conv2d(width * 15 // 8, width, 1, 1, 0),
            *[Init_Conv(width) for _ in range(2)],
            nn.Conv2d(width, in_channel, 3, 1, 1)
        )

    def forward(self, x):
        out1 = self.began(x)  # 1c  1wh
        out1 = self.encoder[0](out1)  # 1c  1wh

        out2 = self.encoder[1](out1)  # 2c  1/2wh
        out2 = self.encoder[2](out2)  # 2c  1/2wh

        out3 = self.encoder[3](out2)  # 4c  1/4wh
        out3 = self.encoder[4](out3)  # 4c  1/4wh

        out4 = self.encoder[5](out3)  # 8c  1/8wh
        out4 = self.main(out4)  # 8c  1/8wh

        up1 = self.up[0](out4)  # 1/8c  1wh
        up2 = self.up[1](out3)  # 1/4c  1wh
        up3 = self.up[2](out2)  # 1/2c  1wh

        out = self.decoder(torch.cat([up1, up2, up3, out1], dim=1))

        return out + x


class MyADeblurModel_M(nn.Module):
    def __init__(self, in_channel=3, pth=None, load=True, decoders_blocks=None):
        super().__init__()
        self.width = 32
        if decoders_blocks is None:
            decoders_blocks = [4, 2, 1]
        self.model1 = MyADeblurModel_V2(in_channel=3, width=32, num_main=50, up_scale=8)
        if load:
            self.model1.load_state_dict(torch.load(pth))
        self.encoder = nn.ModuleList([
            nn.Sequential(self.model1.began[0], self.model1.began[1]),
            nn.Sequential(self.model1.began[2], self.model1.began[3]),
            nn.Sequential(self.model1.began[4], self.model1.began[5])
        ])
        self.main = nn.Sequential(self.model1.began[6], self.model1.began[7], self.model1.main)
        self.decoder = nn.ModuleList([
            UpBlock(self.width * 8),
            nn.Sequential(*[WaveletBlock(self.width * 4) for _ in range(decoders_blocks[0])]),
            UpBlock(self.width * 4),
            nn.Sequential(*[WaveletBlock(self.width * 2) for _ in range(decoders_blocks[1])]),
            UpBlock(self.width * 2),
            nn.Sequential(*[WaveletBlock(self.width) for _ in range(decoders_blocks[2])]),
        ])
        self.end = nn.Sequential(
            nn.Conv2d(self.width, in_channel, 1, 1, 0)
        )
        self.num_layers = self.model1.num_layers + 7

    def forward(self, inp):
        with torch.no_grad():
            en1 = self.encoder[0](inp)  # 1c 1wh
            en2 = self.encoder[1](en1)  # 2c 1/2wh
            en3 = self.encoder[2](en2)  # 4c 1/4wh
            x = self.main(en3)  # 8c 1/8qh

        de3 = self.decoder[0](x)  # 4c 1/4wh
        de3 = self.decoder[1](de3 + en3)  # 4c 1/4wh

        de2 = self.decoder[2](de3)  # 2c 1/2wh
        de2 = self.decoder[3](de2 + en2)  # 2c 1/2wh

        de1 = self.decoder[4](de2)  # 1c 1wh
        de1 = self.decoder[5](de1 + en1)  # 1c 1wh

        out = self.end(de1)
        return out + inp


class MyMltuDeblurModel(nn.Module):
    def __init__(self, in_channel=3, width=32, num_main=20, en_decoders_blocks=None):
        super().__init__()
        self.width = width
        self.num_layers = num_main + 8
        if en_decoders_blocks is None:
            en_decoders_blocks = [1, 1, 1]

        self.began = nn.Conv2d(in_channel, width, 3, 1, 1)
        self.encoder = nn.ModuleList([
            nn.Sequential(*[Init_Conv_V3(width) for _ in range(en_decoders_blocks[0])]),
            nn.Conv2d(width, width * 2, 2, 2, 0),

            nn.Sequential(*[Init_Conv_V3(width * 2) for _ in range(en_decoders_blocks[1])]),
            nn.Conv2d(width * 2, width * 4, 2, 2, 0),

            nn.Sequential(*[Init_Conv_V3(width * 4) for _ in range(en_decoders_blocks[2])]),
            nn.Conv2d(width * 4, width * 8, 2, 2, 0),
        ])
        self.fusion = nn.ModuleList([
            UpBlock(width * 8),
            nn.Sequential(
                nn.Conv2d(width * 8, width * 4, 3, 1, 1),
                *[Init_Conv_V3(width * 4) for _ in range(1)]
            ),

            UpBlock(width * 4),
            nn.Sequential(
                nn.Conv2d(width * 4, width * 2, 3, 1, 1),
                *[Init_Conv_V3(width * 2) for _ in range(1)]
            ),

            UpBlock(width * 2),
        ])
        self.main = nn.Sequential(*[Init_Conv_V3(width * 8, num_heads=4) for _ in range(num_main)],)
        self.decoder = nn.ModuleList([
            UpBlock(width * 8),
            nn.Sequential(
                nn.Conv2d(width * 8, width * 4, 3, 1, 1),
                *[Init_Conv_V3(width * 4) for _ in range(en_decoders_blocks[2])]
            ),

            UpBlock(width * 4),
            nn.Sequential(
                nn.Conv2d(width * 4, width * 2, 3, 1, 1),
                *[Init_Conv_V3(width * 2) for _ in range(en_decoders_blocks[1])]
            ),

            UpBlock(width * 2),
            nn.Sequential(
                nn.Conv2d(width * 3, width * 2, 3, 1, 1),
                Init_Conv_V3(width * 2, num_heads=4),
                nn.Conv2d(width * 2, width, 3, 1, 1),
                *[Init_Conv_V3(width) for _ in range(en_decoders_blocks[0])]
            ),
        ])
        self.end = nn.Conv2d(width, in_channel, 3, 1, 1)

    def forward(self, inp):
        x = self.began(inp)  # 1c, 1wh
        # encode stage
        en1 = self.encoder[0](x)  # 1c, 1wh

        en2 = self.encoder[1](en1)  # 2c, 1/2wh
        en2 = self.encoder[2](en2)  # 2c, 1/2wh

        en3 = self.encoder[3](en2)  # 4c, 1/4wh
        en3 = self.encoder[4](en3)  # 4c, 1/4wh

        en4 = self.encoder[5](en3)  # 8c, 1/8wh
        en4 = self.main(en4)  # 8c, 1/8wh

        # fusion feature
        fu3 = self.fusion[0](en4)  # 4c, 1/4wh
        fu3 = self.fusion[1](torch.cat([fu3, en3], dim=1))  # 4c, 1/4wh

        fu2 = self.fusion[2](fu3)  # 2c, 1/2wh
        fu2 = self.fusion[3](torch.cat([fu2, en2], dim=1))  # 2c, 1/2wh

        fu1 = self.fusion[4](fu2)  # 1c, 1wh

        # decode stage
        de3 = self.decoder[0](en4)  # 4c, 1/4wh
        de3 = self.decoder[1](torch.cat([de3, fu3], dim=1))  # 4c, 1/4wh

        de2 = self.decoder[2](de3)  # 2c, 1/2wh
        de2 = self.decoder[3](torch.cat([de2, fu2], dim=1))  # 2c, 1/2wh

        de1 = self.decoder[4](de2)  # 1c, 1wh
        de1 = self.decoder[5](torch.cat([de1, fu1, en1], dim=1))  # 1c, 1wh

        out = self.end(de1)

        return out + inp


class SP_Deblur(nn.Module):
    def __init__(self, in_channel=3, width=32, num_main=20):
        super(SP_Deblur, self).__init__()
        self.width = width
        self.num_layers = num_main
        self.began = nn.Conv2d(in_channel, width, 1, 1, 0)
        self.main = nn.Sequential(*[SP_block(width, num_heads=4) for _ in range(num_main)])
        self.end = nn.Conv2d(width, in_channel, 1, 1, 0)

    def forward(self, inp):
        x = self.began(inp)
        x = self.main(x)
        x = self.end(x)

        return inp + x


if __name__ == '__main__':
    from torchinfo import summary
    from ptflops import get_model_complexity_info
    from torchvision.models import mobilenet_v3_small

    device = torch.device('cuda:0')

    # model = RDDeblurModel(width=32, num_main=3, up_scale=8)
    # model1_path = '../result/weight/MyADeblurModel_V2_W32L53_best_psnr(31.47_dropout0.2).pth'
    # model = MyADeblurModel_M(in_channel=3, pth=model1_path).to(device)
    model = SP_Deblur(in_channel=3, width=32, num_main=10).to(device)
    # print(mobilenet_v3_small())
    summary(model, (8, 3, 256, 256), col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"))
    # macs, params = get_model_complexity_info(models, (3, 256, 256), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
