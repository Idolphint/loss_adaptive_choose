import numpy as np
from torch.utils.data.dataset import Dataset
from utils import linear_gray_transpose
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
import torch
import sys
import cv2
import os
import xlrd
def ReadXlsx(Fileroot, sheet_num):
    workbook = xlrd.open_workbook(Fileroot)
    sheet = workbook.sheet_by_index(sheet_num)
    return sheet
class UNetTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, is_train=True):

        sheet = ReadXlsx( '/home/tingyang/HeartVessel/GroupIndex/' + 'VolumIndex20200505.xlsx', 1)
        files = sheet.col_values(0)

        row_num = sheet.nrows
        all_img_paths = []
        all_mask_paths = []

        count = 1
        for order in range(0,row_num):
            path1 = os.path.join(img_dir, files[order])
            path2 = os.path.join(mask_dir, files[order])
            if( os.path.exists(path1) and os.path.exists(path2) ):
                all_img_paths.append(path1)
                all_mask_paths.append(path2)

        shuffled_indices = np.arange(0, row_num )
        np.random.shuffle( shuffled_indices )

        self.train_num =  len(all_img_paths) 
        print( 'train num : ', self.train_num)

        train_indices = shuffled_indices
        
        indices = train_indices
        self.img_paths, self.mask_paths = [], []
        for idx in indices:
            self.img_paths.append(all_img_paths[idx])
            self.mask_paths.append(all_mask_paths[idx])
        print(img_dir, len(self.img_paths) )
        print(mask_dir, len(self.mask_paths) )

    def __getitem__(self, index):

        mask = np.array(loadmat(self.mask_paths[index] , verify_compressed_data_integrity=False )['data'])
        img_array = np.array(loadmat(self.img_paths[index] , verify_compressed_data_integrity=False )['data'])

        img = linear_gray_transpose(img_array)
    
        # img = img.astype(np.float32) / 255.
        return torch.Tensor(img[np.newaxis,:,:]), torch.Tensor(mask)

    def __len__(self):
        return self.train_num

if __name__ == '__main__':
    dataset = UNetTestDataset('data/train', 'data/testlabel')
    img, mask = dataset[0]
    print(img.type(), mask.type())

