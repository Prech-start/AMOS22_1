import os.path
import pickle
import matplotlib.pyplot as plt
import numpy as np


def save_loss(t_loss, v_loss):
    t_loss = np.array(t_loss, dtype=np.float)
    v_loss = np.array(v_loss, dtype=np.float)
    with open(os.path.join('tem.tmp'), 'wb+') as f:
        pickle.dump(np.array([t_loss, v_loss]), f)
    pass


def pic_loss_line():
    with open(os.path.join('tem.tmp'), 'rb+') as f:
        loss_ = pickle.load(f)
        len_train = len(loss_[0])
        train_loss, valid_loss = loss_[0], loss_[1]
        plt.plot([i for i in range(len_train)], train_loss, '-', label='train_loss')
        plt.plot([i for i in range(len_train)], valid_loss, '-', label='valid_loss')
        plt.legend()
        plt.savefig('loss_line_final.png', bbox_inches='tight')
    pass

