import torch
from torch import optim
import torch.nn as nn
from unet import UNet
import os
from eval import eval_net
import sys
import scipy.io as scio
import numpy as np
from testdataset import UNetTestDataset
from torch.utils.data.dataloader import DataLoader
import xlrd
from utils import linear_gray_transpose
from scipy.io import loadmat

def ReadXlsx(Fileroot, sheet_num):
    workbook = xlrd.open_workbook(Fileroot)
    sheet = workbook.sheet_by_index(sheet_num)
    return sheet

class UNetTester():
    def __init__(self, dataset, dataloader, model, cfg):
        self.dataset = dataset
        self.dataloader = dataloader
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        #self.device = 'cuda:2'

        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr = cfg.base_lr,
            # momentum = 0.9,
            weight_decay = 0.0005
        )
        test_iter = 63085

        file_name = 'iter_{}.pth'.format(test_iter)
        self.model.load_state_dict(
            torch.load(os.path.join(cfg.checkpoint_dir, file_name))
        )
        self.model.eval()
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        print('begin test :')
        iter_ = 0
        data_dir = cfg.img_dir
        label_dir = cfg.mask_dir

        split_dir = cfg.splitDataset_path

        sheet = ReadXlsx( '/home/yangtingyang/HeartVessel/Dataset/HeartVeesel/GroupIndex/' + 'SelVolum0604.xlsx', 0)
        # sheet = ReadXlsx( '/home/tingyang/HeartVessel/GroupIndex/' + 'TrainSliceIndex0607.xlsx', 0)

        print('Read over')
        
        files = sheet.col_values(0)

        row_num = sheet.nrows
        print('row_num',row_num)
        
        out_dir = './DNW20201002/' #??
        # out_dir = './CEOutputTrain20200830/'

        if( os.path.exists(out_dir) == 0 ):
            os.mkdir(out_dir)

        for name_order in range(0,1):
            for i in range(0, row_num ):
                if( os.path.exists( data_dir + files[i] ) ):
                    img_array = np.array(loadmat( data_dir + files[i] , verify_compressed_data_integrity=False )['data'])
                    mask = np.array(loadmat( label_dir + files[i] , verify_compressed_data_integrity=False )['data'])

                    img = linear_gray_transpose(img_array)
                    
                    imgs = torch.tensor( (np.zeros( (1,1,1,192,192) ) ).astype(np.float32) )
                    true_masks = torch.tensor( (np.zeros( (1,1,192,192)).astype(np.float32) ))
                    imgs[0,:,:,:] = torch.tensor( img )
                    true_masks[0,:,:,:] = torch.tensor( mask )

                    # imgs = np.repeat(imgs, 3, axis=1)
                    img = imgs[0,0,:,:]
                    label = true_masks[0,0,:,:]
                    img = img.numpy()
                    label = label.numpy()

                    imgs = imgs.to(self.device)
                    true_masks = true_masks.to(self.device)

                    masks_pred = self.model(imgs)

                    if(masks_pred.is_cuda==False):
                        masks_pred = masks_pred.to(self.device)

                    n1,c1,w1,h1 = masks_pred.shape
                    n2,c2,w2,h2 = true_masks.shape

                    res = masks_pred[0,:,:,:]
                    res = res.cpu().detach().numpy()

                    masks_pred = torch.transpose( masks_pred,1,2)
                    masks_pred = torch.transpose( masks_pred,2,3)
                    masks_pred = torch.reshape(masks_pred, ( n1*w1*h1 ,c1) )
                    true_masks = true_masks.view(-1)

                    true_masks = true_masks.long()

                    loss = 3*self.criterion(masks_pred, true_masks)
                    res_save_dir = out_dir + files[i]

                    print('iter: {} | loss: {:.7f}'.format(iter_, loss.item()) , res_save_dir)                

                    scio.savemat( res_save_dir , mdict = { 'data' : res } )
                    iter_ = iter_ + 1

