import time
from utils import linear_gray_transpose #[0,1]
from utils.evaluation import *
import torch
import os
import scipy.io as scio
from config import TrainOptions
from unet import UNet
from dataset import UNetDataset

cfg = TrainOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = 'cuda:2'
model = UNet(
    n_channels = 1,
    n_classes = 2
)
model = model.to(device)

data_dir = cfg.img_dir
label_dir = cfg.mask_dir
dataset = UNetDataset('/home/yangtingyang/yty/HeartVessel/Dataset//Resize192Img/', '/home/yangtingyang/yty/HeartVessel/Dataset//Resize192LabelAug/', is_train=False)

checkpointdir = './jobs/'
DNWdir = '/home/yangtingyang/yty/ltt/multi_loss/res/'


def getDNWmat(ckptdir, lossname):
    listnum = []
    foldName = os.path.join(DNWdir, time.strftime('%Y%m%d'))
    os.makedirs(foldName, exist_ok=True)
    for fn in os.listdir(ckptdir):
        spfn = fn.split('.')
        if len(spfn) > 1:
            spfn1 = spfn[0]
         
            num = int( spfn1[5:] )
            listnum.append( num )

    listnum.sort()
    print( listnum )
    Evares = [ ]

    for i in range( 0, len(listnum) ):
        # if( listnum[i] > 60000 ):
        #     break
        Evares.append( [ ] )
        checkpoint = os.path.join(ckptdir, 'iter_' + str( listnum[i] ) + '.pth')
        print(checkpoint)
        model.load_state_dict( torch.load( checkpoint ) )
        model.eval()
        mk= 0
        for img_array, mask in dataset:
            img = linear_gray_transpose(img_array)
            imgs = torch.tensor( (np.zeros( (1,1,192,192) ) ).astype(np.float32) )
            imgs[ 0,:,:,: ] = torch.tensor( img )  #batch = 1
            # imgs = np.repeat(imgs, 3, axis=1)
            imgs = imgs.to( device )
            masks_pred = model(imgs) #n, c, w,h
            res = masks_pred[0,:,:,:] #the first one
        
            res = res.cpu().detach().numpy() #c, w,h
            mask = mask.cpu().detach().numpy()#w,h
            Evares[i].append( [] )
            eva = GetMeanIouDice2( res.copy(), mask.copy(), 0.5 ) #以0.5为阈值，计算iou和dice
            #eva = GetMultiThresholdEva0(res.copy(), mask.copy(), 0.5)
            Evares[i][mk]=eva
            if( mk % 100 == 0 ):
                print( checkpoint, mk,
                       'mIoU={:.3f}, mdice={:.3f}, IoU0={:.3f}, dice0={:.3f}, IoU1={:.3f}, dice1={:.3f}'.format(
                           eva[0], eva[1], eva[2], eva[3], eva[4], eva[5]) ,flush=True)
            mk+=1
        
    Evares = np.array(Evares)
    print("end!!!!")
    scio.savemat( os.path.join(foldName, 'DNW_'+lossname+'.mat') , mdict = { 'data' : Evares } )



# loss_name = ['', 'L2', 'ce', 'focal', 'dice', 'v1']
loss_name = ['Fscore']
for lossi in loss_name:
    print("====================test "+lossi+" ======================")
    ckptlittledir = os.path.join(checkpointdir, lossi)
    getDNWmat(ckptlittledir, lossi)

print("all loss has been evaluated!")

