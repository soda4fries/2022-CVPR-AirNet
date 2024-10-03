import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset
from net.model import AirNet

from option import options as opt
import os
import wandb
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from test import modular_denoising_test

if __name__ == '__main__':
    torch.cuda.set_device(opt.cuda)
    subprocess.check_output(['mkdir', '-p', opt.ckpt_path])
    
    wandb.init(project='WaveletTrain', name="LR_ADJUST", config={
        'learning_rate': opt.lr,
        'epochs': opt.epochs,
        'pretrain_batch_size': opt.batch_size,
    })
    

    trainset = TrainDataset(opt)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    changed = False
    
    #wandb.log({'updated_batch_size': opt.batch_size * 128})
    # Network Construction
    net = AirNet(opt).cuda()
    net.train()

    # Optimizer and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    CE = nn.CrossEntropyLoss().cuda()
    l1 = nn.L1Loss().cuda()
    #ssim_module = SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim = True)
    

    # Start training
    print('Start training...')
    for epoch in range(opt.epochs):
        for ([clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2) in tqdm(trainloader):
            degrad_patch_1, degrad_patch_2 = degrad_patch_1.cuda(), degrad_patch_2.cuda()
            clean_patch_1, clean_patch_2 = clean_patch_1.cuda(), clean_patch_2.cuda()
            

            optimizer.zero_grad()

            if epoch < opt.epochs_encoder:
                _, output, target = net.E(x_query=degrad_patch_1, x_key=degrad_patch_2)
                contrast_loss = CE(output, target)
                loss = contrast_loss
                
            else:
                
                    
                restored, output, target = net(x_query=degrad_patch_1, x_key=degrad_patch_2)
                
                #ssim_loss = 1 - ssim_module(restored, clean_patch_1)
                contrast_loss = CE(output, target)
                l1_loss = l1(restored, clean_patch_1)
                loss = l1_loss + 0.1 * contrast_loss #+ ssim_loss
                

            # backward
            loss.backward()
            optimizer.step()
            
            

        if epoch < opt.epochs_encoder:
            print(
                'Epoch (%d)  Loss: contrast_loss:%0.4f\n' % (
                    epoch, contrast_loss.item(),
                ), '\r', end='')
            
            wandb.log({'epoch': epoch, 'contrast_loss': contrast_loss.item()})
            
            if(epoch == opt.epochs_encoder - 1 and not changed):
                    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
                    # trainloader = DataLoader(trainset, batch_size=opt.batch_size*4, pin_memory=True, shuffle=True,
                    #         drop_last=True, num_workers=4)
                    # changed = True
                    # print("changed")
                    # wandb.log({'updated_batch_size': opt.batch_size * 4})
        else:
            print(
                'Epoch (%d)  Loss: l1_loss:%0.4f contrast_loss:%0.4f\n ssim_loss:%0.4f\n' % (
                    epoch, l1_loss.item(), contrast_loss.item(), 0
                ), '\r', end='')
            
            wandb.log({'epoch': epoch, 'l1_loss': l1_loss.item(), 'contrast_loss': contrast_loss.item(), 'msim_loss': 0})
            
        wandb.log({'learning_rate': optimizer.param_groups[0]['lr']})

        GPUS = 1
        if (epoch + 1) % 50 == 0: 
            psnr_val, ssim_val = modular_denoising_test(net, "/scr/user/mahdinur/CVPR/2022-CVPR-AirNet/test/BSD68/")
            wandb.log({'psnr_val': psnr_val, 'ssim_val' : ssim_val})
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            if GPUS == 1:
                torch.save(net.state_dict(), opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')
            else:
                torch.save(net.module.state_dict(), opt.ckpt_path + 'epoch_' + str(epoch + 1) + '.pth')

        # if epoch <= opt.epochs_encoder:
        #     lr = opt.lr * (0.1 ** (epoch // 60))
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # else:
        #     lr = 0.0001 * (0.5 ** ((epoch - opt.epochs_encoder) // 125))
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
    wandb.finish()
