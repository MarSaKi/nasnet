from controller import Controller
from Worker import Worker
from ProcessWorker import ProcessWorker, get_acc
import torch
import torch.optim as optim
import logging
from multiprocessing import Process, Queue
import time,random

def consume(worker):
    get_acc(worker)
    return worker

class PPO(object):
    def __init__(self, args, train_queue, valid_queue, device):
        self.args = args
        self.device = device

        self.arch_epochs = args.arch_epochs
        self.arch_lr = args.arch_lr
        self.episodes = args.episodes
        self.entropy_weight = args.entropy_weight

        self.ppo_epochs = args.ppo_epochs

        self.controller = Controller(args, device=device).to(device)

        self.adam = optim.Adam(params=self.controller.parameters(), lr=self.arch_lr)

        self.baseline = None
        self.baseline_weight = self.args.baseline_weight

        self.train_queue = train_queue
        self.valid_queue = valid_queue

        self.clip_epsilon = 0.2

    def multi_solve_environment(self):
        for arch_epoch in range(self.arch_epochs):
            workers_queue = Queue()
            results_queue = Queue()

            for episode in range(self.episodes):
                actions_p, actions_log_p, actions_index = self.controller.sample()
                #print(actions_index)
                if 0 < episode < self.episodes // 3:
                    workers_queue.put(ProcessWorker(actions_p.cpu(), actions_log_p.cpu(), actions_index.cpu(), self.args, 'cuda:0'))
                elif self.episodes // 3 <= episode < 2 * self.episodes // 3:
                    workers_queue.put(ProcessWorker(actions_p.cpu(), actions_log_p.cpu(), actions_index.cpu(), self.args, 'cuda:1'))
                else:
                    workers_queue.put(ProcessWorker(actions_p.cpu(), actions_log_p.cpu(), actions_index.cpu(), self.args, 'cuda:3'))

            #consumers = [Process(target=consume, args=(workers_queue, results_queue, self.train_queue, self.valid_queue)) for i in range(self.episodes)]
            consumers = [Process(target=consume, args=(workers_queue, results_queue)) for i in range(self.episodes)]
            [consumers[i].start() for i in range(self.episodes)]
            [consumers[i].join() for i in range(self.episodes)]
            for i, consumer in enumerate(consumers):
                consumer.start()
                print('process {} start'.format(i))
            for i, consumer in enumerate(consumers):
                consumer.join()
                print('process {} finish'.format(i))

            worker = results_queue.get()
            print(worker.genotype)
            workers = [results_queue.get(True) for i in range(self.episodes)]
            print(len(workers))
            for episode in range(self.episodes):
                worker = results_queue.get()
                print(worker.genotype)

            workers = []
            for episode in range(self.episodes):
                worker = results_queue.get()
                worker.actions_p.to(self.device)
            acc = 0
            print(len(workers))

            for episode, worker in enumerate(workers):
                if self.baseline == None:
                    self.baseline = worker.acc
                else:
                    self.baseline = self.baseline * self.baseline_weight + worker.acc * (1 - self.baseline_weight)

                acc += worker.acc
                logging.info('episode {:0>3d} acc {:.4f} baseline {:.4f}'.format(episode, worker.acc, self.baseline))

            acc /= self.episodes
            logging.info('arch_epoch {:0>3d} acc {:.4f} '.format(arch_epoch, acc))

            for ppo_epoch in range(self.ppo_epochs):
                loss = 0

                for worker in workers:
                    actions_p, actions_log_p = self.controller.get_p(worker.actions_index)

                    loss += self.cal_loss(actions_p, actions_log_p, worker, self.baseline)

                loss /= len(workers)
                logging.info('ppo_epoch {:0>3d} loss {:.4f} '.format(ppo_epoch, loss))

                self.adam.zero_grad()
                loss.backward()
                self.adam.step()


    def solve_environment(self):
        for arch_epoch in range(self.arch_epochs):
            workers = []
            acc = 0

            for episode in range(self.episodes):
                actions_p, actions_log_p, actions_index = self.controller.sample()
                workers.append(Worker(actions_p, actions_log_p, actions_index, self.args, self.device))

            for episode, worker in enumerate(workers):
                worker.get_acc(self.train_queue, self.valid_queue)
                if self.baseline == None:
                    self.baseline = worker.acc
                else:
                    self.baseline = self.baseline * self.baseline_weight + worker.acc * (1 - self.baseline_weight)

                acc += worker.acc
                logging.info('episode {:0>3d} acc {:.4f} baseline {:.4f}'.format(episode, worker.acc, self.baseline))
            acc /= self.episodes
            logging.info('arch_epoch {:0>3d} acc {:.4f} '.format(arch_epoch, acc))

            for ppo_epoch in range(self.ppo_epochs):
                loss = 0

                for worker in workers:
                    actions_p, actions_log_p = self.controller.get_p(worker.actions_index)

                    loss += self.cal_loss(actions_p, actions_log_p, worker, self.baseline)

                loss /= len(workers)
                logging.info('ppo_epoch {:0>3d} loss {:.4f} '.format(ppo_epoch, loss))

                self.adam.zero_grad()
                loss.backward()
                self.adam.step()

    def clip(self, actions_importance):
        lower = torch.ones_like(actions_importance).to(self.device) * (1 - self.clip_epsilon)
        upper = torch.ones_like(actions_importance).to(self.device) * (1 + self.clip_epsilon)

        actions_importance, _ = torch.min(torch.cat([actions_importance.unsqueeze(0), upper.unsqueeze(0)], dim=0), dim=0)
        actions_importance, _ = torch.max(torch.cat([actions_importance.unsqueeze(0), lower.unsqueeze(0)], dim=0), dim=0)

        return actions_importance

    def cal_loss(self, actions_p, actions_log_p, worker, baseline):
        actions_importance = actions_p / worker.actions_p
        clipped_actions_importance = self.clip(actions_importance)
        reward = worker.acc - baseline
        actions_reward = actions_importance * reward
        clipped_actions_reward = clipped_actions_importance * reward

        actions_reward, _ = torch.min(torch.cat([actions_reward.unsqueeze(0), clipped_actions_reward.unsqueeze(0)], dim=0), dim=0)
        policy_loss = -1 * torch.sum(actions_reward)
        entropy = -1 * torch.sum(actions_p * actions_log_p)
        entropy_bonus = -1 * entropy * self.entropy_weight

        return policy_loss + entropy_bonus

