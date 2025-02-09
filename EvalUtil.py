import os
import random
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import torch.nn.functional as tf
import torch
from tqdm import tqdm
import time
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, \
                               LearnedPerceptualImagePatchSimilarity

from Deblur_Utils.data.data_load import Val_DeblurDataset
from models.MyDeblurModel.MyAttDeblur import MyADeblurModel, MyADeblurModel_V2, MyMltuDeblurModel
from TrainUtil import test_model


def Val_Model(device, model, data_path, interval, transformer=None, use_half=False):
    began_time = time.time()
    with open(f'./result/val_log/{model.__class__.__name__}_W{model.width}D{model.num_layers}.txt', 'w', encoding='utf-8') as f:
        f.write('=' * 40 + model.__class__.__name__ + 'interval ' + str(interval) +
                f'device={device}' + '=' * 40 + '\n', )
    total_psnr = 0.
    total_ssim = 0.
    total_lpips = 0.

    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    lpips = LearnedPerceptualImagePatchSimilarity().to(device)

    blur_path = os.path.join(data_path, 'blur')
    sharp_path = os.path.join(data_path, 'sharp')

    num_img = len(os.listdir(blur_path)) // interval
    for i, img_name in enumerate(tqdm(os.listdir(blur_path))):
        if i % interval == 0:
            blur_img_path = os.path.join(blur_path, img_name)
            sharp_img_path = os.path.join(sharp_path, img_name)

            blur_img = Image.open(blur_img_path)
            sharp_img = Image.open(sharp_img_path)
            if transformer:
                blur_img = transformer(blur_img)
                sharp_img = transformer(sharp_img)
            img_size = blur_img.size

            blur_img = F.to_tensor(blur_img).unsqueeze(0).to(device)
            sharp_img = F.to_tensor(sharp_img).unsqueeze(0).to(device)

            since = time.time()
            with torch.no_grad():
                if use_half:
                    model.half()
                    blur_img = blur_img.half()
                    sharp_img = sharp_img.half()
                pred = model(blur_img)
                pred = torch.clamp(pred, 0, 1)
            time_cost = time.time() - since

            val_psnr, val_ssim, val_lpips = psnr(pred, sharp_img), ssim(pred, sharp_img), lpips(pred, sharp_img)
            total_psnr += val_psnr.item()
            total_ssim += val_ssim.item()
            total_lpips += val_lpips.item()

            with open(f'./result/val_log/{model.__class__.__name__}_W{model.width}D{model.num_layers}.txt', 'a', encoding='utf-8') as f:
                f.write('img_name:{:.10} | img_size:{:.13} | val_psnr:{:.10} | val_ssim:{:.10} | val_lpips:{:.10} | '
                        'time_cost:{:.10} ms \n'
                        .format(str(img_name).ljust(10), str(img_size).ljust(13), str(val_psnr.item()).ljust(10),
                                str(val_ssim.item()).ljust(10), str(val_lpips.item()).ljust(10), time_cost * 1000))
    total_time = time.time() - began_time
    with open(f'./result/val_log/{model.__class__.__name__}_W{model.width}D{model.num_layers}.txt', 'a', encoding='utf-8') as f:
        f.write('avg_psnr is :{:.10}, avg_ssim is :{:.10}, avg_lpips is :{:.10}, total_time is :{:.10} s'.
                format(total_psnr / num_img, total_ssim / num_img, total_lpips / num_img, total_time))


if __name__ == '__main__':
    from torchinfo import summary
    torch.manual_seed(42)

    device = torch.device('cuda:0')
    # device = torch.device('cpu')

    model = MyADeblurModel_V2(in_channel=3, width=48, num_main=50, num_encoder=1, up_scale=8)
    model.load_state_dict(torch.load('./result/weight/MyADeblurModel_V2_W48L53_best_psnr(31.80).pth'))

    summary(model, (1, 3, 256, 256), col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"))
    print(torch.get_float32_matmul_precision())
    data_path = r'E:\datasets\DeBlur\GoPro\test'

    transform = transforms.CenterCrop(256)
    use_half = False
    model.eval()
    Val_Model(device, model, data_path=data_path, interval=1, use_half=use_half)
    # valid_data = Val_DeblurDataset(r'E:\datasets\DeBlur\GoPro\test\crop_256', crop_size=256, is_crop=False)
    # valid_dataloader = DataLoader(valid_data, batch_size=8, num_workers=4, prefetch_factor=16, pin_memory=True)
    # test_model(device, model, valid_dataloader)

