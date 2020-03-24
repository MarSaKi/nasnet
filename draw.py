import matplotlib.pyplot as plt
import numpy as np

def valid_compare(path1, path2):
    f1 = open(path1, 'r')
    f2 = open(path2, 'r')
    acc1 = []
    acc2 = []

    for line in f1:
        if 'arch_epoch' in line and 'acc' in line:
            print(line)
            line = line.rstrip('\n').split(' ')
            acc_idx = line.index('acc') + 1
            acc = float(line[acc_idx])
            acc1.append(acc)

    for line in f2:
        if 'arch_epoch' in line and 'acc' in line:
            line = line.rstrip('\n').split(' ')
            acc_idx = line.index('acc') + 1
            acc = float(line[acc_idx])
            acc2.append(acc)

    epochs = np.linspace(0, 99, 100)
    plt.figure()
    plt.plot(epochs, acc1, label = 'fixed baseline', color = 'cyan')
    plt.plot(epochs, acc2, label = 'dynamic baseline', color = 'orange')
    plt.legend(loc = 'best')
    plt.savefig('fig/valid_compare.pdf')

valid_compare('log/fix_0.95.txt', 'log/dynamic_b.txt')