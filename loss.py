import torch
import torch.nn as nn
from config import TrainOptions
import torch.nn.functional as F
from torch.autograd import Variable
import os

#Variable is a simple wrapper of tensor ,with recording of creator and grad .. 

cfg = TrainOptions().parse()

class MultiLosses(object):
    def __init__(self, weight=None, cuda=True, device=cfg.device):
        self.cuda = cuda
        self.device = torch.device(device)
        self.weight = weight
        self.loss = self.L1loss

    def bulid_loss(self, mode='ce'):
        """ choices: [0: L1, 1: L2, 2: exp, 3: ce, 4: focal, 5: dice, 6:smooth01]
        """
        if mode == 'L2' or mode == '0':
            self.loss = self.MSEloss
        elif mode == 'ce' or mode == '1':
            self.loss = self.crossentropy
        elif mode == 'focal' or mode == '2':
            self.loss = self.focal1
        elif mode == 'dice' or mode == '3':
            self.loss = self.MultiClassDiceloss
        elif mode == 'Fscore' or mode == '4':
            self.loss = self.softFMeasure
        #elif mode == 'smooth01' or mode == '6':
        #    self.loss = self.smooth01loss
        else:
            raise NotImplementedError

    def toOneHot(self, label, class_num = 10):
        ''' input [n, x,x], output [n, c, x, x]
        '''
        n, w, h = label.shape
        flush = torch.zeros(n, class_num, w, h).to(self.device)

        for i in range(n):
            for j in range(class_num):
                flush[i, j , :,:] = torch.eq(label[i,:,:], j)
        one_hot = flush.contiguous()

        return one_hot

    ##是否应该考虑把c领出来？
    def L1loss(self, logit, target):
        pred = logit.view(-1)
        one_hot_target = self.toOneHot(target, class_num = cfg.class_num) #n, c, w, hkeneng zhkanl
        groundT = one_hot_target.view(-1)
        #? cuda????
        return torch.mean(torch.abs(pred - groundT))

    def MSEloss(self, logit, target):
        pred = logit.view(-1)
        one_hot_target = self.toOneHot(target, class_num = cfg.class_num)
        groundT = one_hot_target.view(-1)
        return torch.mean(torch.pow(pred - groundT, 2)) 

    def exploss(self, logit, target):
        pred = logit.contiguous().view(-1)
        groundT = target.contiguous().view(-1)

        return torch.mean(torch.exp(-1 * groundT * pred))

    def multiClassExp(self, logit, target):
        n, c, w, h = logit.shape
        OH_target = self.toOneHot(target, class_num = cfg.class_num)
        exp_total = 0
        for i in range(c):
            exp_item = self.exploss(logit[:,i,:,:], OH_target[:,i,:,:])
            exp_total += exp_item
        return exp_total / c

    def crossentropy(self, logit, target):

        n, c, h, w = logit.size()
        # logit = F.softmax(logit, dim=1)
        criterion = nn.CrossEntropyLoss(weight=self.weight)

        loss = criterion(logit, target)
        return loss

    def focalloss(self, logit, target): #only for 2 class!!
        alpha = 1.0
        gamma = 2
        n, c, h, w = logit.size()
        # logit = F.softmax(logit, dim=1)
        criterion = nn.CrossEntropyLoss(weight=self.weight)

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)

        #logpt *= alpha

        loss = -((1-pt) ** gamma) *logpt


        return loss
    def focal1(self, logit, target): #no use ce
        gamma = 2.0
        N, C, W, H = logit.shape
        P = F.softmax(logit, dim=1)

        class_mask = self.toOneHot(target, class_num = cfg.class_num)

        probs = (P*class_mask).sum(1).view(-1,1) #sum.dim=1
        log_p = probs.log()

        batch_loss = -(torch.pow((1-probs), gamma))*log_p
        
        loss = batch_loss.mean() * 5  #all loss mean about pixel, so just *10
        return loss



    def diceloss(self, logit, target):
        ##是否需要将target转为 onehot
        logit = logit.contiguous().view(-1)
        target = target.contiguous().view(-1)

        smooth = 0.0001
        intersection = torch.sum(target * logit)

        dice = (2. * intersection + smooth) / (torch.sum(target * target) +
         torch.sum(logit * logit) + smooth)

        return (1. - dice)
        
    def MultiClassDiceloss(self, logit, target):
        n, c, w, h = logit.size()
        OHtarget = self.toOneHot(target, class_num = cfg.class_num)
        dice_loss = 0
        for i in range(c):
            dice_loss += self.diceloss(logit[:,i,:,:], OHtarget[:,i,:,:])
        
        return dice_loss / c
    def normalData(self, data):
        maxp = torch.max(data)
        minp = torch.min(data)
        nor_data = (data - minp) / (maxp - minp)
        return nor_data

    def softFMeasure(self, logit, target, beta=1):

        # logit = logit.contiguous().view(-1)
        # target = target.contiguous().view(-1)
        logit_nor = self.normalData(logit)
        OHtarget = self.toOneHot(target, class_num=cfg.class_num)

        zh_target = OHtarget[:,0,:,:]
        zh_logit = logit_nor[:,0,:,:]
        re_target = OHtarget[:,1,:,:]
        re_logit = logit_nor[:,1,:,:]

        TP = torch.sum(zh_logit * zh_target)
        FP = torch.sum(zh_logit * re_target)
        FN = torch.sum(re_logit * zh_target)

        FM_b = ((1+beta*beta)*TP) / ( (1+beta*beta)*TP + beta*beta*FN + FP)
        loss = 1. - FM_b

        return loss

    def smooth01loss(self, logit, target):
        pass

    def forward(self):
        return self.loss

    def getBiggestGrad(self):
        #TODO
        pass


if __name__ == "__main__":
    criterion = MultiLosses()
    a= torch.rand(2,2,192,192).to(criterion.device)
    b = torch.rand(2,192,192).to(criterion.device).long()
    criterion.bulid_loss(mode ='ce')
    print(criterion.loss(a,b).item())
    for i in range(7):
        mode_str = str(i)
        criterion.bulid_loss(mode = mode_str)
        print(criterion.loss(a,b).item())



