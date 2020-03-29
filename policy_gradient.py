from controller import Controller
from Worker import Worker, get_acc
import torch
import torch.optim as optim
import numpy as np
import logging
from multiprocessing import Process, Queue
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def consume(worker, results_queue):
    get_acc(worker)
    results_queue.put(worker)

class PolicyGradient(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.arch_epochs = args.arch_epochs
        self.arch_lr = args.arch_lr
        self.episodes = args.episodes
        self.entropy_weight = args.entropy_weight

        self.controller = Controller(args, device=device).to(device)

        self.adam = optim.Adam(params=self.controller.parameters(), lr=self.arch_lr)

        self.baseline = None
        self.baseline_weight = self.args.baseline_weight

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
                #worker.actions_p = torch.Tensor(worker.actions_p).to(self.device)
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
            logging.info(
                'arch_epoch {:0>3d} top1_acc {:.4f} top5_avg_acc {:.4f} top20_avg_acc {:.4f} baseline {:.4f} '.format(
                    arch_epoch, top1_acc, top5_avg_acc, top20_avg_acc, self.baseline))
            for i in range(5):
                print(workers_top20[i].genotype)

            loss = 0
            for worker in workers:
                actions_p, actions_log_p = self.controller.get_p(worker.actions_index)

                loss += self.cal_loss(actions_p, actions_log_p, worker, self.baseline)

            loss /= len(workers)

            self.adam.zero_grad()
            loss.backward()
            self.adam.step()

    def cal_loss(self, actions_p, actions_log_p, worker, baseline):
        reward = worker.acc - baseline
        policy_loss = -1 * torch.sum(actions_log_p * reward)
        entropy = -1 * torch.sum(actions_p * actions_log_p)
        entropy_bonus = -1 * entropy * self.entropy_weight

        return policy_loss + entropy_bonus