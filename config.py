import argparse

class TrainOptions:
    def __init__(self):
        parser = argparse.ArgumentParser()
            
        parser.add_argument('--img_dir', type=str, default='/home/yangtingyang/yty/HeartVessel/Dataset/Resize192Img/')
        parser.add_argument('--mask_dir', type=str, default='/home/yangtingyang/yty/HeartVessel/Dataset/Resize192LabelAug/')
        
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--version', type=int, default=2)
        parser.add_argument('--base_lr', type=float, default=0.00004) #16 : 0.00004, 8 : 0.00002
        parser.add_argument('--epoch', type=int, default=30)
        parser.add_argument('--display_freq', type=int, default=30)
        parser.add_argument('--eval_freq', type=int, default=300)
        parser.add_argument('--class_num', type=int, default=2)
        parser.add_argument('--start_iter', type=int, default=0)
        parser.add_argument('--job_path', type=str, default='./jobs')
        parser.add_argument('--save_freq', type=int, default=120)
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--use_strategy', action='store_true')
        parser.add_argument('--checkpoint_dir', type=str, default='./jobs')
        parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--device', type=str, default='cuda:2')
        parser.add_argument('--split_train_Dataset_path', type=str, default='/home/yangtingyang/LTT/DenseNet/splitRecord.txt')
        parser.add_argument('--split_test_Dataset_path', type=str, default='/home/yangtingyang/LTT/DenseNet/splitRecord2.txt')
        self.parser = parser

    def parse(self):
        opt = self.parser.parse_args()
        return opt
