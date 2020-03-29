from controller import Controller
from Worker import Worker, get_acc
import numpy as np
import torch
import torch.optim as optim
import logging
from multiprocessing import Process, Queue
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def consume(worker, results_queue):
    get_acc(worker)
    results_queue.put(worker)

class PPO(object):
    def __init__(self, args, device):
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

        self.clip_epsilon = 0.2

    def multi_solve_environment(self):
        workers_top20 = []

        for arch_epoch in range(self.arch_epochs):
            results_queue = Queue()
            processes = []

            for episode in range(self.episodes):
                actions_p, actions_log_p, actions_index = self.controller.sample()
                actions_p = actions_p.cpu().numpy().tolist()
                actions_log_p = actions_log_p.cpu().numpy().tolist()
                actions_index = actions_index.cpu().numpy().tolist()

                if episode < self.episodes // 3:
                    worker = Worker(actions_p, actions_log_p, actions_index, self.args, 'cuda:0')
                elif self.episodes // 3 <= episode < 2 * self.episodes // 3:
                    worker = Worker(actions_p, actions_log_p, actions_index, self.args, 'cuda:1')
                else:
                    worker = Worker(actions_p, actions_log_p, actions_index, self.args, 'cuda:3')

                process = Process(target=consume, args=(worker, results_queue))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

            workers = []
            for episode in range(self.episodes):
                worker = results_queue.get()
                worker.actions_p = torch.Tensor(worker.actions_p).to(self.device)
                worker.actions_index = torch.LongTensor(worker.actions_index).to(self.device)
                workers.append(worker)

            for episode, worker in enumerate(workers):
                if self.baseline == None:
                    self.baseline = worker.acc
                else:
                    self.baseline = self.baseline * self.baseline_weight + worker.acc * (1 - self.baseline_weight)

            # sort worker retain top20
            workers_total = workers_top20 + workers
            workers_total.sort(key=lambda worker: worker.acc, reverse=True)
            workers_top20 = workers_total[:20]
            top1_acc = workers_top20[0].acc
            top5_avg_acc = np.mean([worker.acc for worker in workers_top20[:5]])
            top20_avg_acc = np.mean([worker.acc for worker in workers_top20])
            logging.info('arch_epoch {:0>3d} top1_acc {:.4f} top5_avg_acc {:.4f} top20_avg_acc {:.4f} baseline {:.4f} '.format(
                arch_epoch, top1_acc, top5_avg_acc, top20_avg_acc, self.baseline))
            for i in range(5):
                print(workers_top20[i].genotype)

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

