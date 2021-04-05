import torch
from torch.utils.data.dataloader import DataLoader
import os
from config import TrainOptions
from unet import UNet
from dataset import UNetDataset
from testdataset import UNetTestDataset

from trainer import UNetTrainer
from tester import UNetTester

import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if __name__ == '__main__':
    cfg = TrainOptions().parse()
    print('lr: ', cfg.base_lr)
    unet_dataset = UNetTestDataset(
        img_dir = cfg.img_dir,
        mask_dir = cfg.mask_dir,
        is_train = False
    )
    unet_test_dataset = UNetTestDataset(
        img_dir = cfg.img_dir,
        mask_dir = cfg.mask_dir,
        is_train = False
    )
    unet_dataloader = DataLoader(
        dataset = unet_dataset,
        batch_size = 1,
        shuffle = True,
        num_workers = cfg.num_workers
    )
    model = UNet(
        n_channels = 1,
        n_classes = 2
    )
    tester = UNetTester(
        dataset = unet_test_dataset,
        dataloader = unet_dataloader,
        model = model,
        cfg = cfg
    )

    try:
        tester.train()
    except KeyboardInterrupt:
        # torch.save(model.state_dict(), os.path.join(cfg.job_path, 'iter_0.pth'))
        # print('Saved iter_0.pth')
        print( 'test end' )
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)



