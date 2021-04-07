import torch
from torch.utils.data.dataloader import DataLoader
import os
from config import TrainOptions
from unet import UNet
from dataset import UNetDataset
from trainer import UNetTrainer
import sys


if __name__ == '__main__':
    cfg = TrainOptions().parse()
    print('lr: ', cfg.base_lr,flush=True)
    unet_dataset = UNetDataset(
        img_dir = cfg.img_dir,
        mask_dir = cfg.mask_dir,
        is_train = True
    )
    
    #unet_test_dataset = UNetDataset(
    #    img_dir = cfg.img_dir,
    #    mask_dir = cfg.mask_dir,
    #    is_train = False
    #)
    unet_dataloader = DataLoader(
        dataset = unet_dataset,
        batch_size = cfg.batch_size,
        shuffle = True,
        num_workers = cfg.num_workers
    )
    model = UNet(
        n_channels = 1,
        n_classes = 2
    )
    trainer = UNetTrainer(
        dataset = unet_dataset, #unet_test_dataset,
        dataloader = unet_dataloader,
        model = model,
        cfg = cfg
    )
    #loss_list = ['L2', 'ce', 'focal', 'dice', 'Fscore']
    loss_list = ['Fscore']
    for lossi in loss_list:
        print("======================using loss: {} =========================\n".format(lossi))
        try:
            trainer.train(lossi)
        except KeyboardInterrupt:
            torch.save(model.state_dict(), os.path.join(cfg.job_path, 'iter_0.pth'))
            print('Saved iter_0.pth')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)



