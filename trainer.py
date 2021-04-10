import torch
from torch import optim
import torch.nn as nn
from unet import UNet
from torch.autograd import Variable
import os
import numpy as npy
import scipy.io as scio
from loss import MultiLosses

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
class UNetTrainer():
    def __init__(self, dataset, dataloader, model, cfg):
        self.dataset = dataset
        self.dataloader = dataloader
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr = cfg.base_lr,
            # momentum = 0.9,
            weight_decay = 0.0005
        )
        self.loss_function = MultiLosses()
        if cfg.continue_train and cfg.start_iter:
        # cfg.start_iter = 3600
            file_name = 'iter_{}.pth'.format(cfg.start_iter)
            self.model.load_state_dict(
                torch.load(os.path.join(cfg.checkpoint_dir, file_name))
            )

            print('load {} over '.format(file_name))
        self.model.train()
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = DiceCELoss()


    def train(self, loss_name='ce'):
        iter_ = self.cfg.start_iter
        loss_record = []
        lossR_cnt = 0
        print('start_iter: {}'.format(iter_))
        os.makedirs(os.path.join(self.cfg.job_path, loss_name), exist_ok=True)
        for i in range(self.cfg.epoch):

            loss_count = 0

            iter_loss = 0
            iter_num = 0
            iter_count = 0
            num_count = 0
            idx = 0
            maxgrad = 0
            lossChange = 0
            lossi = []
            preLoss = []
            if( i % 5 == 0):
                print('----------------------------- epoch : ', i, ' ---------------------------')
            for data in self.dataloader:
                iter_ += 1
                self.optimizer.zero_grad()
                imgs, true_masks = data

                imgs = imgs.to(self.device)
                imgs.requires_grad_()
                print(imgs.requires_grad)
                true_masks = true_masks.to(self.device)

                masks_pred = self.model(imgs)
                if(masks_pred.is_cuda==False):
                    masks_pred = masks_pred.to(self.device)
                # masks_pred.retain_grad()
                # masks_pred = Variable(masks_pred, requires_grad=True)
                # # 0.4以后variable已经和tensor合并，叶子只需要tensor.requires_grad_()
                # 即可设置为求导，非叶子节点，需要tensor.retain_grad()
                # n1,c1,w1,h1 = masks_pred.shape
                # n2,w2,h2 = true_masks.shape
                # ##### ----------------------- Loss ----------------------
                lossi = []
                maxgrad = 0
                if self.cfg.use_strategy:
                    for lit in range(4):
                        self.loss_function.bulid_loss(str(lit))
                        loss = self.loss_function.loss(masks_pred, true_masks)

                        if self.cfg.version == 2:
                            if len(preLoss) != 0:
                                lossChange = (loss - preLoss[lit]) / preLoss[lit]

                            if (len(preLoss) == 0 or
                                    (lossChange > 0 and lossChange > maxgrad) or
                                    (lossChange < 0 and maxgrad < 0 and lossChange < maxgrad)): #更新要选择的loss
                                maxgrad = lossChange
                                idx = lit
                        elif self.cfg.version == 1:
                            if loss > maxgrad:
                                maxgrad = loss
                                idx = lit
                        elif self.cfg.version == 3:
                            if len(preLoss) != 0:
                                lossChange = (loss - preLoss[lit]) / preLoss[lit]
                            if (len(preLoss) == 0 or abs(lossChange) > maxgrad):
                                maxgrad = abs(lossChange)
                                idx = lit
                        lossi.append(loss.item())
                    preLoss = lossi
                    if iter_ % self.cfg.display_freq == 0:
                        loss_record.append([lossi, idx])
                    self.loss_function.bulid_loss(str(idx))
                else:
                    self.loss_function.bulid_loss(loss_name)
                loss = self.loss_function.loss(masks_pred, true_masks)
                
                loss.backward()
                # print(imgs.grad)
                self.optimizer.step()

                loss_count = loss_count + loss.item()
                num_count = num_count + 1
                iter_loss = iter_loss + loss.item()
                iter_count = iter_count + 1
                if iter_ % self.cfg.display_freq == 0:
                    print('iter: {} | loss: {:.7f}'.format(iter_, iter_loss/iter_count ), flush=True )
                    if self.cfg.use_strategy:
                        print("\t L2: {:.2f}, ce: {:.2f}, focal: {:.2f}, dice: {:.2f}".format(lossi[0],lossi[1], lossi[2],lossi[3]))
                        print("\t criteria: {:.2f}, using loss {}".format(maxgrad, idx))

            loss_average = loss_count/num_count
            print('epoch: {} | loss: {:.7f}'.format(i, loss_average), flush=True)
            file_name = 'iter_{}.pth'.format(iter_)
            save_path = os.path.join(self.cfg.job_path, loss_name, file_name)
            torch.save(self.model.state_dict(), save_path)
            print('save {} over!'.format(save_path))
        
        file_name = 'iter_{}.pth'.format(iter_)
        save_path = os.path.join(self.cfg.job_path, loss_name, file_name)
        torch.save(self.model.state_dict(), save_path)
        print('save {} over!'.format(save_path))

        loss_record = npy.array(loss_record)
        print(loss_record.shape)
        scio.savemat("./lossRecord-{}.mat".format(loss_name), mdict ={'data': loss_record})


