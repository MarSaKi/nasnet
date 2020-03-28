from controller import Controller
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import utils
from model import Network
import torchvision
import numpy as np
from torch.autograd import Variable
from collections import deque
import logging
from multiprocessing import Pool

class PolicyGradient(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.arch_epochs = args.arch_epochs
        self.arch_lr = args.arch_lr
        self.episodes = args.episodes
        self.entropy_weight = args.entropy_weight
        self.init_baseline = args.init_baseline

        self.controller = Controller(args, device=device).to(device)
        logging.info('controller param size = {}MB'.format(utils.count_params(self.controller)))
        self.adam = optim.Adam(params=self.controller.parameters(), lr=self.arch_lr)

        self.baseline = None
        self.baseline_weight_decay = self.args.baseline_weight_decay

    def solve_environment(self, train_queue, valid_queue):
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        max_valid_acc = 0
        max_genotype = None

        for arch_epoch in range(self.arch_epochs):
            acc = 0
            loss = 0
            entropy = 0
            avg_valid_acc = np.mean(self.valid_accs)
            #avg_valid_acc = self.init_baseline

            # one thread
            for episode in range(self.episodes):
                ps, log_ps, actions_index = self.controller.sample()

                genotype = utils.parse_actions_index(actions_index)
                print(genotype)
                valid_acc = self.get_valid_acc(genotype)
                if valid_acc > max_valid_acc:
                    max_valid_acc = valid_acc
                    max_genotype = genotype

                reward = valid_acc - avg_valid_acc
                self.valid_accs.append(valid_acc)

                episode_loss, episode_entropy = self.cal_loss(ps, log_ps, reward)
                logging.info('episode {:0>3d} acc {:.4f} avg_acc {:.4f} reward {:.4f} loss {:.4f} entropy {:.4f}'.format(
                    episode, valid_acc, avg_valid_acc, reward, float(episode_loss), float(episode_entropy)
                ))

                acc += valid_acc
                loss += episode_loss
                entropy += episode_entropy

            # multi thread
            '''list_ps = []
            list_log_ps = []
            list_actions_index = []
            list_genotype = []
            for episode in range(self.episodes):
                ps, log_ps, actions_index = self.controller.sample()
                list_ps.append(ps)
                list_log_ps.append(log_ps)
                list_actions_index.append(actions_index)
                list_genotype.append(utils.parse_actions_index(actions_index))

            pool = multiprocessing.Pool(12)
            list_valid_acc = pool.map(self.get_valid_acc, list_genotype)
            pool.close()
            pool.join()

            if max(list_valid_acc) > max_valid_acc:
                max_valid_acc = max(list_valid_acc)
                max_idx = list_valid_acc.index(max(list_valid_acc))
                max_genotype = list_genotype[max_idx]

            avg_valid_acc = np.mean(self.valid_accs)
            for episode in range(self.episodes):
                reward = list_valid_acc[episode] - avg_valid_acc
                self.valid_accs.append(list_valid_acc[episode])

                episode_loss, episode_entropy = self.cal_loss(list_ps[episode], list_log_ps[episode], reward)
                logging.info(
                    'episode {:0>3d} acc {:.4f} avg_acc {:.4f} reward {:.4f} loss {:.4f} entropy {:.4f}'.format(
                        episode, list_valid_acc[episode], avg_valid_acc, reward, float(episode_loss), float(episode_entropy)
                    ))

                acc += list_valid_acc[episode]
                loss += episode_loss
                entropy += episode_entropy'''

            acc /= self.episodes
            loss /= self.episodes
            entropy /= self.episodes
            logging.info('arch_epoch {:0>3d} acc {:.4f} loss {:.4f} entropy {:.4f}'.format(
                arch_epoch, acc, float(loss), float(entropy)))
            logging.info('{:.4f} {}'.format(max_valid_acc, max_genotype))

            self.adam.zero_grad()
            loss.backward()
            self.adam.step()

    def cal_loss(self, ps, log_ps, reward):
        policy_loss = -1 * torch.sum(log_ps) * reward
        entropy = -1 * torch.sum(ps * log_ps)

        #to maxium entropy and policy reward
        loss = policy_loss - self.entropy_weight * entropy
        return loss, entropy

    def cal_reward(self, acc):
        if self.baseline == None:
            self.baseline = acc
        else:
            self.baseline = self.baseline_weight_decay * self.baseline + (1-self.baseline_weight_decay) * acc
        reward = acc - self.baseline
        return reward

    def get_valid_acc(self, genotype):
        criterion = nn.CrossEntropyLoss()
        model = Network(genotype).to(self.device)

        optimizer = torch.optim.SGD(model.parameters(),
                                    self.args.model_lr,
                                    momentum=self.args.model_momentum,
                                    weight_decay=self.args.model_weight_decay)

        for model_epoch in range(self.args.model_epochs):
            train_loss, train_acc = train(model, self.train_queue, criterion, optimizer, self.device)
            #print('train loss {:.4f} acc {:.4f}'.format(train_loss, train_acc))

        valid_loss, valid_acc = infer(model, self.valid_queue, criterion, self.device)
        #print('valid loss {:.4f} acc {:.4f}'.format(valid_loss, valid_acc))

        return valid_acc

def train(model, train_queue, creterion, optimizer, device):
    avg_loss = 0
    avg_acc = 0
    batch_num = len(train_queue)

    model.train()
    for batch, (input, target) in enumerate(train_queue):
        input = Variable(input, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)

        optimizer.zero_grad()
        logits = model(input)
        loss = creterion(logits, target)
        loss.backward()
        optimizer.step()

        acc = utils.accuracy(logits.data, target.data)[0]
        avg_loss += float(loss)
        avg_acc += float(acc)

    return avg_loss / batch_num, avg_acc / batch_num

def infer(model, valid_queue, creterion, device):
    avg_loss = 0
    avg_acc = 0
    batch_num = len(valid_queue)

    model.eval()
    for batch, (input, target) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input).to(device)
            target = Variable(target).to(device)

            logits = model(input)
            loss = creterion(logits, target)
        acc = utils.accuracy(logits.data, target.data)[0]
        avg_loss += float(loss)
        avg_acc += float(acc)

    return avg_loss / batch_num, avg_acc / batch_num