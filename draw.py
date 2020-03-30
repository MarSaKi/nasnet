import matplotlib.pyplot as plt
import numpy as np

def get_acc(path):
    f = open(path, 'r')
    top1_acc = []
    top5_acc = []
    top20_acc = []
    for line in f:
        if 'arch_epoch' in line and 'top1_acc' in line:
            print(line)
            line = line.rstrip('\n').split(' ')
            top1_acc.append(float(line[line.index('top1_acc')+1]))
            top5_acc.append(float(line[line.index('top5_avg_acc') + 1]))
            top20_acc.append(float(line[line.index('top20_avg_acc') + 1]))
    return top1_acc[:100], top5_acc[:100], top20_acc[:100]

def draw_acc(RS_path, PG_path, PPO_path, out_path):
    rs_top1, rs_top5, rs_top20 = get_acc(RS_path)
    pg_top1, pg_top5, pg_top20 = get_acc(PG_path)
    ppo_top1, ppo_top5, ppo_top20 = get_acc(PPO_path)
    epochs = np.linspace(0, 99, 100)

    plt.figure()
    plt.plot(epochs, rs_top1, label='RS top1 acc', color='green')
    plt.plot(epochs, rs_top5, label='RS top5 acc', linestyle='-.', color='green')
    plt.plot(epochs, rs_top20, label='RS top20 acc', linestyle='--', color='green')

    plt.plot(epochs, pg_top1, label='PG top1 acc', color='blue')
    plt.plot(epochs, pg_top5, label='PG top5 acc', linestyle='-.', color='blue')
    plt.plot(epochs, pg_top20, label='PG top20 acc', linestyle='--', color='blue')

    plt.plot(epochs, ppo_top1, label='PPO top1 acc', color='red')
    plt.plot(epochs, ppo_top5, label='PPO top5 acc', linestyle='-.', color='red')
    plt.plot(epochs, ppo_top20, label='PPO top20 acc', linestyle='--', color='red')

    plt.legend(loc='lower right')
    plt.title('Validation Accuracy on Mnist')
    plt.xlabel('Search Epoch')
    plt.ylabel('Accuracy at 5 Epochs')
    plt.savefig(out_path)

draw_acc('log/train_RS.log', 'log/train_PG.log', 'log/train_PPO.log', 'fig/search.png')