import numpy as np
from operations import *
from genotypes import Genotype
import os
import shutil

def count_params(model):
    return sum(np.prod(v.shape) for name,v in model.named_parameters())/1e6

def parse_actions_index(actions_index):
    normal = []
    reduce = []
    normal_concat = set(range(2,6))
    reduce_concat = set(range(2,6))

    for i in range(8):
        node1 = int(actions_index[i*5])
        node2 = int(actions_index[i*5+1])

        op1 = OP_NAME[actions_index[i*5+2]]
        op2 = OP_NAME[actions_index[i*5+3]]

        comb = COMB_NAME[actions_index[i*5+4]]

        block = (node1, node2, op1, op2, comb)
        if i < 4:
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
