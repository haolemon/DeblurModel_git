import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import time
from torchinfo import summary
from models.NAFNet.NAFNet_arch import NAFNet
from models.NAFNet.Baseline_arch import Baseline
from models.MIMO_Unet import MIMOUNet
# from MLWNet.models.MLWNet_arch import MLWNet
from models.MyDeblurModel.MyAttDeblur import MyADeblurModel, MyADeblurModel_V2, MyADeblurModel_V3, MyMltuDeblurModel, \
                                       MyADeblurModel_V4, MyADeblurModel_M, SP_Deblur
from Deblur_Utils.data.data_load import DeblurDataset, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor, \
    PairCompose, Val_DeblurDataset, SSID_LMDB_Read, PairCenterCrop, RealBlurDataset
from TrainUtil import train, valid
from Deblur_Utils.losses import PSNRLoss, L1_Charbonnier_loss
from torchvision import models
from torch.utils import tensorboard
from datetime import datetime
import timm


def main():
    # device = torch.device('cpu')
    device = torch.device('cuda:0')

    enc_blks = [1, 1, 1, 28]
    dec_blks = [1, 1, 1, 1]
    model1_path = './result/weight/MyADeblurModel_V2_W32L53_best_psnr(31.47_dropout0.2).pth'
    # model = MyADeblurModel_M(pth=model1_path).to(device)
    # 迁移学习
    model = MyADeblurModel_V2(in_channel=3, width=32, num_main=30).to(device)
    # model = NAFNet(img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    # model.load_state_dict(torch.load('result/weight/MyADeblurModel_V2_W32L53_best_psnr(31.47_dropout0.2).pth'))
    # model.expend_deep(10)
    # models.load_state_dict(torch.load('./deblur_test/Pretrained.pth'))
    model_info = summary(model, (8, 3, 256, 256),
                         col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"))
    model_param = model_info.total_params
    model_mult_adds = model_info.total_mult_adds
    inp_size = model_info.input_size
    # setattr(model, 'num_layers', 48)
    # setattr(model, 'width', 32)
    print('layers: ', model.num_layers)

    lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3, amsgrad=False, betas=(0.9, 0.9))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2103 * 100, eta_min=1e-7)  # 1116 2103 1879
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_func = PSNRLoss().to(device)
    # loss_func = L1_Charbonnier_loss().to(device)
    # loss_func = nn.L1Loss().to(device)
    # loss_func = nn.MSELoss().to(device)

    batch_size = 8
    num_workers = 4
    prefetch_factor = 16
    train_transform = PairCompose(
        [
            PairRandomCrop(256),
            PairRandomHorizontalFilp(),
            PairToTensor()
        ]
    )
    valid_transform = PairCompose(
        [
            PairCenterCrop(256),
            PairToTensor()
        ]
    )
    # E:\datasets\DeBlur\SIDD_Small\train\crop_512   E:\python_work\CV-learning\NAFNet\datasets\GoPro\train
    train_data1 = DeblurDataset(r'E:\datasets\DeBlur\GoPro\train\crop_512', transform=train_transform,
                                expend_scale=1)
    # RealBlur
    # train_data1 = RealBlurDataset(r'E:\datasets\DeBlur\RealBlur\RealBlur',
    #                               txt_path='./dataset/RealBlur/RealBlur_J_train_list.txt', transform=train_transform,
    #                               expend_scale=4)

    # train_data2 = DeblurDataset(r'E:\datasets\DeBlur\GoPro\test\crop_512', train_transform, expend_scale=1)
    # train_data = ConcatDataset([train_data1, train_data2])
    train_dataloader = DataLoader(train_data1, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  prefetch_factor=prefetch_factor, pin_memory=True)
    # GoPro
    valid_data = Val_DeblurDataset(r'E:\datasets\DeBlur\GoPro\test\crop_256', crop_size=256, is_crop=False)
    valid_dataloader = DataLoader(valid_data, batch_size=8, num_workers=num_workers,
                                  prefetch_factor=prefetch_factor, pin_memory=True)
    # SSID
    # valid_data = SSID_LMDB_Read('./dataset/SSID/Val')
    # valid_dataloader = DataLoader(valid_data, batch_size=batch_size, num_workers=0, pin_memory=True)

    # RealBlur
    # valid_data = RealBlurDataset(r'E:\datasets\DeBlur\RealBlur\RealBlur',
    #                              txt_path='./dataset/RealBlur/RealBlur_J_test_list.txt', transform=valid_transform)
    # valid_dataloader = DataLoader(valid_data, batch_size=8, num_workers=2, prefetch_factor=16, pin_memory=True)

    epochs = 100
    use_half = False
    mult_out = False

    with open(f'result/train_log/{model.__class__.__name__}_log_W{model.width}_L{model.num_layers}.txt', 'a',
              encoding='utf-8') as f:
        f.write(f'训练的模型为: {model.__class__.__name__}W{model.width}L{model.num_layers} '
                f'参数：{model_param / 1e6}M  计算量：{model_mult_adds / 1e9}G 输入形状：{inp_size}\n')

    began = time.time()
    PSNR_Best = 0.
    Loss_Best = 99.

    # torch.compile(models)
    # block_idx = 24
    # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    # writer1 = tensorboard.SummaryWriter(log_dir=f'./board/{TIMESTAMP}_main{block_idx}_conv1_weight')
    # writer2 = tensorboard.SummaryWriter(log_dir=f'./board/{TIMESTAMP}_main{block_idx}_conv2_weight')
    for epoch in range(epochs):
        since = time.time()
        # 权重分布分析
        # conv1 = model.main[block_idx].conv1.weight
        # conv2 = model.main[block_idx].conv2.weight
        # writer1.add_histogram(f'main{block_idx}_conv1_weight', conv1, global_step=epoch)
        # writer2.add_histogram(f'main{block_idx}_conv2_weight', conv2, global_step=epoch)

        # 训练
        train(device, model, train_dataloader, optimizer, loss_func, scheduler, epoch, epochs, fre_print=265,
              use_half=use_half, amp=True)
        # 验证
        loss, psnr = valid(device, model, valid_dataloader, loss_func, epoch, epochs, use_half=use_half)

        print('One Epoch Cost Time Is :', time.time() - since)

        if psnr > PSNR_Best:
            torch.save(model.state_dict(),
                       f'./result/weight/{model.__class__.__name__}_W{model.width}L{model.num_layers}_best_psnr.pth')
            PSNR_Best = psnr

        # if loss < Loss_Best:
        #     torch.save(models.state_dict(),
        #                f'result/weight/{models.__class__.__name__}_W{models.width}L{models.num_layers}_best_loss.pth')
        #     Loss_Best = loss

        # torch.save(models.state_dict(),
        #            f'result/weight/{models.__class__.__name__}_W{models.width}L{models.num_layers}_now.pth')

    end_time = time.time()
    print('Train Finish, Total Cost Time Is :', (end_time - began) // 60, '分', (end_time - began) % 60, '秒')
    print()


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    main()
