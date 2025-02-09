import torch
from torch import nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm
import Deblur_Utils.down_core as down_core
from torch.cuda.amp import autocast, GradScaler


class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(1e-8)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


def train(device, model, dataloader, optimizer, loss_func, scheduler, epoch, num_epochs, fre_print=500,
          use_half=False, amp=False):
    train_psnr = PeakSignalNoiseRatio().to(device)
    train_ssim = StructuralSimilarityIndexMeasure().to(device)

    l1_func = nn.L1Loss().to(device)
    scaler = GradScaler()

    psnr_epoch = Adder()
    ssim_epoch = Adder()
    loss_epoch = Adder()

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                        unit="batch")
    model.train()
    if use_half:
        model.half()
    for batch_idx, batch_data in progress_bar:
        blur_img = batch_data[0].to(device).float()
        sharp_img = batch_data[1].to(device).float()
        if use_half:
            blur_img = blur_img.half()
            sharp_img = sharp_img.half()

        if amp:
            with autocast():
                pred_img = model(blur_img)
                loss = loss_func(pred_img, sharp_img)
                #
                # optimizer.zero_grad()
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                # scheduler.step()
        else:
            pred_img = model(blur_img)
            loss = loss_func(pred_img, sharp_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            psnr = train_psnr(pred_img, sharp_img)
            ssim = train_ssim(pred_img, sharp_img)

        psnr_epoch(psnr.item())
        ssim_epoch(ssim.item())
        loss_epoch(loss.item())

        progress_bar.set_postfix(loss=loss.item(), psnr=psnr.item(), ssim=ssim.item(),
                                 lr=optimizer.param_groups[0]['lr'])
        if batch_idx % fre_print == 0 and batch_idx != 0:
            tqdm.write(f"Iter {batch_idx} :, Train Loss: {loss_epoch.average():.7f}, "
                       f"Train PSNR: {psnr_epoch.average():.7f}, Train SSIM: {ssim_epoch.average():.7f}")

            torch.save(model.state_dict(),
                       f'result/weight/{model.__class__.__name__}_W{model.width}L{model.num_layers}_now.pth')

            with open(f'result/train_log/{model.__class__.__name__}_log_W{model.width}_L{model.num_layers}.txt', 'a',
                      encoding='utf-8') as f:
                f.write(f'Iter {batch_idx} : Train Loss: {loss_epoch.average():.5f},'
                        f'Train PSNR: {psnr_epoch.average():.5f}, Train SSIM: {ssim_epoch.average():.5f} \n')
            psnr_epoch.reset(), ssim_epoch.reset(), loss_epoch.reset()
    # scheduler.step()

    tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_epoch.average():.7f}, "
               f"Train PSNR: {psnr_epoch.average():.7f}, Train SSIM: {ssim_epoch.average():.7f}")
    with open(f'result/train_log/{model.__class__.__name__}_log_W{model.width}_L{model.num_layers}.txt', 'a', encoding='utf-8') as f:
        f.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_epoch.average():.5f},'
                f'Train PSNR: {psnr_epoch.average():.5f}, Train SSIM: {ssim_epoch.average():.5f} \n \n')


def valid(device, model, dataloader, loss_func, epoch, num_epochs, use_half=False):
    valid_psnr = PeakSignalNoiseRatio().to(device)
    valid_ssim = StructuralSimilarityIndexMeasure().to(device)

    psnr_epoch = 0.
    ssim_epoch = 0.
    loss_epoch = 0.
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                        unit="batch")

    model.eval()
    if use_half:
        model.half()
    num_sampler = len(dataloader)
    for batch_idx, batch_data in progress_bar:
        blur_img = batch_data[0].to(device)
        sharp_img = batch_data[1].to(device)
        if use_half:
            blur_img = blur_img.half()
            sharp_img = sharp_img.half()

        with torch.no_grad():
            pred_img = model(blur_img)
            loss = loss_func(pred_img, sharp_img)

            psnr = valid_psnr(pred_img, sharp_img)
            ssim = valid_ssim(pred_img, sharp_img)

            psnr_epoch += psnr.item()
            ssim_epoch += ssim.item()
            loss_epoch += loss.item()

        progress_bar.set_postfix(psnr=psnr.item(), ssim=ssim.item())

    tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, "
               f"Valid PSNR: {psnr_epoch/num_sampler:.7f}, Valid SSIM: {ssim_epoch/num_sampler:.7f}")
    with open(f'result/train_log/{model.__class__.__name__}_log_W{model.width}_L{model.num_layers}.txt', 'a', encoding='utf-8') as f:
        f.write(f'Epoch {epoch + 1}/{num_epochs}, '
                f'Valid PSNR: {psnr_epoch/num_sampler:.5f}, Valid SSIM: {ssim_epoch/num_sampler:.5f} \n\n')

    return loss_epoch/num_sampler, psnr_epoch/num_sampler


def test_model(device, model, dataloader, use_half=False, mult_out=False):
    valid_psnr = PeakSignalNoiseRatio().to(device)
    valid_ssim = StructuralSimilarityIndexMeasure().to(device)

    psnr_epoch = 0.
    ssim_epoch = 0.
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), unit="batch")

    model.eval()
    if use_half:
        model.half()
    num_sampler = len(dataloader)
    for batch_idx, batch_data in progress_bar:
        blur_img = batch_data[0].to(device)
        sharp_img = batch_data[1].to(device)
        if use_half:
            blur_img = blur_img.half()
            sharp_img = sharp_img.half()

        with torch.no_grad():
            if mult_out:
                sharp_img2, sharp_img4 = down_core.imresize(sharp_img, scale=1 / 2), down_core.imresize(sharp_img, scale=1 / 4)
                pred_img, pred_img2, pred_img4 = model(blur_img)
            else:
                pred_img = model(blur_img)
                # pred_img = blur_img
            psnr = valid_psnr(pred_img, sharp_img)
            ssim = valid_ssim(pred_img, sharp_img)

            psnr_epoch += psnr.item()
            ssim_epoch += ssim.item()

        progress_bar.set_postfix(psnr=psnr.item(), ssim=ssim.item())

    tqdm.write( f"Valid PSNR: {psnr_epoch/num_sampler:.7f}, Valid SSIM: {ssim_epoch/num_sampler:.7f}")
    # with open(f'result/val_log/{models.__class__.__name__}_log_W{models.width}_L{models.num_layers}.txt', 'a', encoding='utf-8') as f:
    #     f.write(f'Valid PSNR: {psnr_epoch/num_sampler:.5f}, Valid SSIM: {ssim_epoch/num_sampler:.5f} \n\n')
