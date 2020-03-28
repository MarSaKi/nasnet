import torch
import torch.nn as nn
import utils
from model import Network
from torch.autograd import Variable
import torchvision
import torch.backends.cudnn as cudnn
import numpy as np

class Worker(object):
    def __init__(self, actions_p, actions_log_p, actions_index, args, device):
        self.actions_p = actions_p
        self.actions_log_p = actions_log_p
        self.actions_index = actions_index
        self.genotype = utils.parse_actions_index(actions_index)

        self.args = args
        self.device = device

        self.params_size = None
        self.acc = None

def get_acc(worker):
    torch.manual_seed(worker.args.seed)
    np.random.seed(worker.args.seed)
    if torch.cuda.is_available():
        device = torch.device(worker.device)
        cudnn.benchmark = True
        cudnn.enable = True
        torch.cuda.manual_seed(worker.args.seed)
    else:
        device = torch.device('cpu')

    train_transform, valid_transform = utils._data_transforms_cifar10(worker.args)
    train_data = torchvision.datasets.MNIST(root=worker.args.data, train=True,
                                            transform=train_transform,
                                            download=True)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(worker.args.train_portion * num_train))
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=worker.args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=worker.args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    model = Network(worker.genotype).to(device)
    worker.params_size = utils.count_params(model)

    optimizer = torch.optim.SGD(model.parameters(),
                                worker.args.model_lr,
                                momentum=worker.args.model_momentum,
                                weight_decay=worker.args.model_weight_decay)

    for model_epoch in range(worker.args.model_epochs):
        train_loss, train_acc = train(model, train_queue, criterion, optimizer, device)
        #print('train loss {:.4f} acc {:.4f}'.format(train_loss, train_acc))

    valid_loss, valid_acc = infer(model, valid_queue, criterion, device)
    print('valid loss {:.4f} acc {:.4f}'.format(valid_loss, valid_acc))

    worker.acc = valid_acc

def train(model, train_queue, criterion, optimizer, device):
    avg_loss = 0
    avg_acc = 0
    batch_num = len(train_queue)

    model.train()
    for batch, (input, target) in enumerate(train_queue):
        input = Variable(input, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        acc = utils.accuracy(logits.data, target.data)[0]
        avg_loss += float(loss)
        avg_acc += float(acc)

    return avg_loss / batch_num, avg_acc / batch_num

def infer(model, valid_queue, criterion, device):
    avg_loss = 0
    avg_acc = 0
    batch_num = len(valid_queue)

    model.eval()
    for batch, (input, target) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input).to(device)
            target = Variable(target).to(device)

            logits = model(input)
            loss = criterion(logits, target)
        acc = utils.accuracy(logits.data, target.data)[0]
        avg_loss += float(loss)
        avg_acc += float(acc)

    return avg_loss / batch_num, avg_acc / batch_num