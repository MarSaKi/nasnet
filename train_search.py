from policy_gradient import PolicyGradient
from PPO import PPO
import utils

import numpy as np
import torch.backends.cudnn as cudnn
import torch
import torchvision

import argparse
import logging
import time
import glob
import os
import sys

parser = argparse.ArgumentParser('minst')
#data
parser.add_argument('--data', type=str, default='./mnist')
parser.add_argument('--train_portion', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
#model
parser.add_argument('--model_epochs', type=int, default=2)
parser.add_argument('--model_lr', type=float, default=0.025)
parser.add_argument('--model_lr_min', type=float, default=0.001)
parser.add_argument('--model_weight_decay', type=float, default=3e-4)
parser.add_argument('--model_momentum', type=float, default=0.9)
parser.add_argument('--init_channel', type=int, default=4)
#architecture
parser.add_argument('--arch_epochs', type=int, default=300)
parser.add_argument('--arch_lr', type=float, default=3.5e-4)
parser.add_argument('--episodes', type=int, default=1)
parser.add_argument('--entropy_weight', type=float, default=1e-5)
parser.add_argument('--baseline_weight', type=float, default=0.95)
parser.add_argument('--embedding_size', type=int, default=32)
parser.add_argument('--algorithm', type=str, choices=['PPO','PG'], default='PPO')
#PPO
parser.add_argument('--ppo_epochs', type=int, default=20)
parser.add_argument('--clip_epsilon', type=float, default=0.2)

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()

exp_dir = 'search_{}_{}'.format(args.algorithm, time.strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(exp_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    logging.info('args = %s', args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(str(args.gpu)))
        cudnn.benchmark = True
        cudnn.enable = True
        logging.info('using gpu : {}'.format(args.gpu))
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
        logging.info('using cpu')

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = torchvision.datasets.MNIST(root=args.data, train=True,
                                            transform=train_transform,
                                            download=True)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=False, num_workers=2)

    '''policy_gradient = PolicyGradient(args, device)
    policy_gradient.solve_environment(train_queue, valid_queue)'''
    ppo = PPO(args, train_queue, valid_queue, device)
    ppo.multi_solve_environment()

main()