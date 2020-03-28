import torch
import torch.nn as nn
import utils
from model import Network
from torch.autograd import Variable

class Worker(object):
    def __init__(self, actions_p, actions_log_p, actions_index, args, device='cpu'):
        self.actions_p = actions_p
        self.actions_log_p = actions_log_p
        self.actions_index = actions_index
        self.genotype = utils.parse_actions_index(actions_index)

        self.args = args
        self.device = device

        self.params_size = None
        self.acc = None

    def get_acc(self, train_queue, valid_queue):
        criterion = nn.CrossEntropyLoss()
        model = Network(self.genotype).to(self.device)
        self.params_size = utils.count_params(model)

        optimizer = torch.optim.SGD(model.parameters(),
                                    self.args.model_lr,
                                    momentum=self.args.model_momentum,
                                    weight_decay=self.args.model_weight_decay)

        for model_epoch in range(self.args.model_epochs):
            train_loss, train_acc = self.trainer(model, train_queue, criterion, optimizer)
            # print('train loss {:.4f} acc {:.4f}'.format(train_loss, train_acc))

        valid_loss, valid_acc = self.inferer(model, valid_queue, criterion)
        # print('valid loss {:.4f} acc {:.4f}'.format(valid_loss, valid_acc))

        self.acc = valid_acc

    def trainer(self, model, train_queue, criterion, optimizer):
        avg_loss = 0
        avg_acc = 0
        batch_num = len(train_queue)

        model.train()
        for batch, (input, target) in enumerate(train_queue):
            input = Variable(input, requires_grad=False).to(self.device)
            target = Variable(target, requires_grad=False).to(self.device)

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            acc = utils.accuracy(logits.data, target.data)[0]
            avg_loss += float(loss)
            avg_acc += float(acc)

        return avg_loss / batch_num, avg_acc / batch_num

    def inferer(self, model, valid_queue, criterion):
        avg_loss = 0
        avg_acc = 0
        batch_num = len(valid_queue)

        model.eval()
        for batch, (input, target) in enumerate(valid_queue):
            with torch.no_grad():
                input = Variable(input).to(self.device)
                target = Variable(target).to(self.device)

                logits = model(input)
                loss = criterion(logits, target)
            acc = utils.accuracy(logits.data, target.data)[0]
            avg_loss += float(loss)
            avg_acc += float(acc)

        return avg_loss / batch_num, avg_acc / batch_num