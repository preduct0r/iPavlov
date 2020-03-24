import os
import pickle
import matplotlib
from matplotlib import pyplot as plt

def plot_loss(experiments_path):
    data_path = os.path.join(experiments_path, "loss_track.pkl")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    [train_loss, valid_loss] = data

    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()