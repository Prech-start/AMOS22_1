import os.path
import pickle
import matplotlib.pyplot as plt
import numpy as np


def save_loss(t_loss, v_loss, v_acc, filename='tem'):
    t_loss = np.array(t_loss, dtype=np.float)
    v_loss = np.array(v_loss, dtype=np.float)
    v_acc = np.array(v_acc, dtype=np.float)
    with open(os.path.join('{}.tmp'.format(filename)), 'wb+') as f:
        pickle.dump(np.array([t_loss, v_loss, v_acc]), f)


def pic_loss_line():
    with open(os.path.join('tem.tmp'), 'rb+') as f:
        loss_ = pickle.load(f)
        len_train = len(loss_[0])
        train_loss, valid_loss, valid_acc = loss_[0], loss_[1], loss_[2]
        plt.plot([i for i in range(len_train)], train_loss, '-', label='train_loss')
        plt.plot([i for i in range(len_train)], valid_loss, '-', label='valid_loss')
        plt.legend()
        plt.savefig('loss_line_final.png', bbox_inches='tight')


def pic_loss_acc(filename='tem'):
    with open(os.path.join('{}.tmp'.format(filename)), 'rb+') as f:
        loss_ = pickle.load(f)
        len_train = len(loss_[0])
        train_loss, valid_loss, valid_acc = loss_[0], loss_[1], loss_[2]

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        lin1 = ax.plot([i for i in range(len_train)], train_loss, '-', label='train_loss', color='blue')
        lin2 = ax.plot([i for i in range(len_train)], valid_loss, '-', label='valid_loss', color='green')

        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax2.set_ylabel('dice')

        lin3 = ax2.plot([i for i in range(len_train)], valid_acc, '-', label='dice', color='black')
        lins = lin1 + lin2 + lin3
        labs = [l.get_label() for l in lins]

        plt.figure(figsize=(8, 8), dpi=100)
        plt.legend(lins, labs, loc='best')
        plt.savefig('{}.png'.format(filename), bbox_inches='tight')


# pic_loss_acc()
