import torch
from torch.utils.data.dataloader import DataLoader
import os
from config import TrainOptions
from unet import UNet
from dataset import UNetDataset
from trainer import UNetTrainer
import sys
import deeplabv3

def load_model(opts):
    model_map = {
        'deeplabv3_resnet50': deeplabv3.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': deeplabv3.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': deeplabv3.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': deeplabv3.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': deeplabv3.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': deeplabv3.deeplabv3plus_mobilenet,
        'unet': UNet
    }

    if 'deeplab' in opts.model:
        model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        if 'plus' in opts.model:
            deeplabv3.convert_to_separable_conv(model.classifier)
    else:
        model = model_map[opts.model](n_channels=1, n_classes=2)
    return model

if __name__ == '__main__':
    cfg = TrainOptions().parse()
    print('lr: ', cfg.base_lr,flush=True)
    my_dataset = UNetDataset(
        img_dir = cfg.img_dir,
        mask_dir = cfg.mask_dir,
        is_train = True
    )
    
    #unet_test_dataset = UNetDataset(
    #    img_dir = cfg.img_dir,
    #    mask_dir = cfg.mask_dir,
    #    is_train = False
    #)
    my_dataloader = DataLoader(
        dataset = my_dataset,
        batch_size = cfg.batch_size,
        shuffle = True,
        num_workers = cfg.num_workers
    )
    model = load_model(cfg)
    trainer = UNetTrainer(
        dataset = my_dataset, #my_test_dataset,
        dataloader = my_dataloader,
        model = model,
        cfg = cfg
    )
    #loss_list = ['L2', 'ce', 'focal', 'dice', 'Fscore']
    loss_list = ['ce']
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



