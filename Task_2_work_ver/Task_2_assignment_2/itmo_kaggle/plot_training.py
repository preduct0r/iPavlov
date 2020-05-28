import numpy as np
import matplotlib.pyplot as plt
import pickle


n_epoch = 100
epochs = np.arange(1, n_epoch+1)

with open(r'C:\Users\Andrey\Downloads\itmo_kaggle\data_from_training.pkl', 'rb') as f:
    [train_loss, train_acc, val_loss, val_acc] = pickle.load(f)

axes = plt.gca()
axes.set_ylim([0,max(max(train_loss), max(val_loss))+1])
plt.plot(epochs, train_loss, 'r', label='train')
plt.plot(epochs, val_loss, 'b', label='val')
plt.title('loss')
plt.xlabel('epoch number')
plt.ylabel('loss value')
plt.legend()
plt.grid()
plt.show()


axes = plt.gca()
axes.set_ylim([0,1])
plt.plot(epochs, train_acc, 'r', label='train')
plt.plot(epochs, val_acc, 'b', label='val')
plt.title('accuracy')
plt.xlabel('epoch number')
plt.ylabel('accuracy value')
plt.legend()
plt.grid()
plt.show()