import numpy as np
from operations import *
from genotypes import Genotype
import os
import shutil
import torch
import torchvision.transforms as transforms

def count_params(model):
    return sum(np.prod(v.shape) for name,v in model.named_parameters())/1e6

def parse_actions_index(actions_index):
    steps = 4
    normal = []
    reduce = []
    normal_concat = set(range(2,6))
    reduce_concat = set(range(2,6))

    for i in range(2*steps):
        node1 = int(actions_index[i*5])
        node2 = int(actions_index[i*5+1])

        op1 = OP_NAME[actions_index[i*5+2]]
        op2 = OP_NAME[actions_index[i*5+3]]

        comb = COMB_NAME[actions_index[i*5+4]]

        block = (node1, node2, op1, op2, comb)
        if i < steps:
            if node1 in normal_concat:
                normal_concat.remove(node1)
            if node2 in normal_concat:
                normal_concat.remove(node2)
            normal.append(block)
        else:
            if node1 in reduce_concat:
                reduce_concat.remove(node1)
            if node2 in reduce_concat:
                reduce_concat.remove(node2)
            reduce.append(block)

    genotype = Genotype(normal = normal, normal_concat = normal_concat,
                        reduce = reduce, reduce_concat = reduce_concat)

    return genotype

def accuracy(logits, target, topk=(1,)):
    assert logits.shape[0]==target.shape[0]
    batch_size = logits.shape[0]
    result = []
    maxk = max(topk)
    target = target.view(-1,1)
    _, pred = torch.topk(logits, maxk, 1, True, True)

    for k in topk:
        predk = pred[:,:k]
        targetk = target.expand_as(predk)
        correct = torch.eq(predk, targetk)
        correct_num = torch.sum(torch.sum(correct, 1),0)
        result.append(float(correct_num)/batch_size)

    return result

def one_hot(index, num_classes):
    v = torch.zeros((num_classes), dtype=torch.float)
    v[int(index)] = 1
    return v

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.5]
  CIFAR_STD = [0.25]

  train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform
