import numpy as np
from torch.utils.data.dataset import Dataset
from utils import linear_gray_transpose
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
from config import TrainOptions
import xlrd
import torch
import sys
import cv2
import os
DEBUG = False
def ReadXlsx(Fileroot, sheet_num):
    workbook = xlrd.open_workbook(Fileroot)
    sheet = workbook.sheet_by_index(sheet_num)
    return sheet


class UNetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, is_train=True):
        print('Dataset')
        cfg = TrainOptions().parse()
        sheet_train = "TrainSliceIndex0607.xlsx"
        sheet_test = "TrainSliceIndex0607.xlsx"
        train_s = 0
        test_s = 2

        sheet_name = sheet_train
        s = train_s
        if is_train == False:
            sheet_name = sheet_test
            s = test_s
        sheet = ReadXlsx( '/home/yangtingyang/yty/HeartVessel/Dataset/GroupIndex/' + sheet_name, s) # test set
        

        row_num = sheet.nrows
        col_num = sheet.ncols
        
        print("sheet has {} rows".format(row_num))

        self.imgs = sheet.col_values(0)

        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __getitem__(self, index):

        cfg = TrainOptions().parse()
        fn = self.imgs[index] #fn 是记录一个包里所有矩阵的文件名的 数组
        imi = loadmat(os.path.join(self.img_dir, fn),
              verify_compressed_data_integrity=False)['data']
        labeli = loadmat(os.path.join(self.mask_dir, fn), 
              verify_compressed_data_integrity=False)['data']
        msk_img = (imi >=0)
          
        img = torch.from_numpy(imi)
        label = torch.from_numpy(labeli)
        img = torch.unsqueeze(img, 0).float() #add channel weiDu
        label = label.long()
        if DEBUG:
            print(img.size(), label.size())
            #label = torch.squeeze(label).long()
        return img,label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    dataset = UNetDataset('/home/yangtingyang/yty/HeartVessel/Dataset/Resize192Img/', '/home/yangtingyang/yty/HeartVessel/Dataset/Resize192LabelAug/', is_train=True)
    img, mask = dataset[4]
    print(img.type(), mask.type())
    print(len(dataset))

