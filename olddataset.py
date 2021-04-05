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
DEBUG = True
def ReadXlsx(Fileroot, sheet_num):
    workbook = xlrd.open_workbook(Fileroot)
    sheet = workbook.sheet_by_index(sheet_num)
    return sheet


class UNetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, is_train=True):
        print('Dataset')
        cfg = TrainOptions().parse()
        sheet_train = ReadXlsx( '/home/yangtingyang/HeartVessel/Dataset/HeartVeesel/GroupIndex/' + 'TrainSliceIndex0607.xlsx', 0) # train set
        sheet_test = ReadXlsx( '/home/yangtingyang/HeartVessel/Dataset/HeartVeesel/GroupIndex/' + 'SelVolum0604.xlsx', 0) # test set
        sheet = sheet_train
        split_path = cfg.split_train_Dataset_path

        if is_train == False:
            sheet = sheet_test
            split_path = cfg.split_test_Dataset_path
        record = open(split_path, 'w')
    
        fh = sheet.col_values(0)
        row_num = sheet.nrows


        imgs = []
        imgbag = []
        labelbag = []
        cnt = 0
        personN = 0
        i=0
        lastinstance = "1000000"
        if DEBUG:
            print("len img list:", row_num)
        while i < row_num:
            lineimg = fh[i].strip('\n')
            body, fix = lineimg.split(".")
            gro, _,_, code, instance = body.split("_") 

            if abs(int(instance) - int(lastinstance)) >= 15: #如果本人的ct图有明显跳变（缺少10帧以上）
              personN = code
              if len(imgbag) != 0: 
                imgs.append(imgbag) #将该组打包上交，不与下面同处一组
                record.write(" "+str(len(imgbag))+"\n")
                imgbag = []
                cnt=0

            elif cnt >= cfg.depth or code != personN: #满则打包
              if code == personN and i>=cfg.depth:
                i -= (cfg.depth-3)
              personN = code
              
              if len(imgbag) != 0:
                imgs.append(imgbag)
                record.write(" "+str(len(imgbag))+"\n")
              imgbag = []
              cnt = 0
            else: #加到该组后面
              imgbag.append(lineimg)
              record.write(lineimg+" ")
              i+=1
              cnt+=1
            lastinstance = instance

        self.imgs = imgs
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        record.close()

    def __getitem__(self, index):

        cfg = TrainOptions().parse()
        fn = self.imgs[index] #fn 是记录一个包里所有矩阵的文件名的 数组
        imgdata = np.zeros((cfg.depth, 192, 192))
        labeldata = np.zeros((cfg.depth, 192, 192))
        if len(fn) == 0:
          print("\n\n\n\n\nitem is all zero \n\n\n\n")
        for i in range(len(fn)):
          imi = loadmat(os.path.join(self.img_dir, fn[i]),
              verify_compressed_data_integrity=False)['data']
          labeli = loadmat(os.path.join(self.mask_dir, fn[i]), 
              verify_compressed_data_integrity=False)['data']
          msk_img = (imi >=0)
          
          imgdata[i, msk_img] = imi[msk_img]
          labeldata[i] = labeli
        while i <cfg.depth:
          imgdata[i] = imi
          labeldata[i] = labeli
          i+=1

        img = torch.from_numpy(imgdata)
        label = torch.from_numpy(labeldata)
        img = torch.unsqueeze(img, 0).float() #add channel weiDu
        label = label.long()
        if DEBUG:
            print(img.size(), label.size())
            #label = torch.squeeze(label).long()
        return img,label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    dataset = UNetDataset('/home/yangtingyang/HeartVessel/Dataset/HeartVeesel/Dataset/Resize192Img/', '/home/yangtingyang/HeartVessel/Dataset/HeartVeesel/Dataset/Resize192LabelAug/', is_train=True)
    img, mask = dataset[4]
    print(img.type(), mask.type())
    print(len(dataset))

