from Worker import Worker, get_acc
import numpy as np
import logging
from multiprocessing import Process, Queue
import random
from operations import *
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def consume(worker, results_queue):
    get_acc(worker)
    results_queue.put(worker)

class RandomSearch(object):
    def __init__(self, args):
        self.args = args

        self.arch_epochs = args.arch_epochs
        self.episodes = args.episodes

    def random_sample(self):
        steps = 4
        len_nodes = steps + 1
        len_OPS = len(OP_NAME)
        len_combs = len(COMB_NAME)
        nodes = list(range(len_nodes))
        OPS = list(range(len_OPS))
        combs = list(range(len_combs))

        actions_index = []

        for type in range(2):
            for node in range(steps):
                actions_index.append(random.choice(nodes[:node+2]))
                actions_index.append(random.choice(nodes[:node+2]))
                actions_index.append(random.choice(OPS))
                actions_index.append(random.choice(OPS))
                actions_index.append(random.choice(combs))

        return actions_index

    def multi_solve_environment(self):
        workers_top20 = []

        for arch_epoch in range(self.arch_epochs):
            results_queue = Queue()
            processes = []

            for episode in range(self.episodes):
                actions_index = self.random_sample()

                if episode < self.episodes // 3:
                    worker = Worker(None, None, actions_index, self.args, 'cuda:0')
                elif self.episodes // 3 <= episode < 2 * self.episodes // 3:
                    worker = Worker(None, None, actions_index, self.args, 'cuda:1')
                else:
                    worker = Worker(None, None, actions_index, self.args, 'cuda:3')

                process = Process(target=consume, args=(worker, results_queue))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

            workers = []
            for episode in range(self.episodes):
                worker = results_queue.get()
                workers.append(worker)

            # sort worker retain top20
            workers_total = workers_top20 + workers
            workers_total.sort(key=lambda worker: worker.acc, reverse=True)
            workers_top20 = workers_total[:20]
            top1_acc = workers_top20[0].acc
            top5_avg_acc = np.mean([worker.acc for worker in workers_top20[:5]])
            top20_avg_acc = np.mean([worker.acc for worker in workers_top20])
            logging.info(
                'arch_epoch {:0>3d} top1_acc {:.4f} top5_avg_acc {:.4f} top20_avg_acc {:.4f}'.format(
                    arch_epoch, top1_acc, top5_avg_acc, top20_avg_acc))
            for i in range(5):
                print(workers_top20[i].genotype)