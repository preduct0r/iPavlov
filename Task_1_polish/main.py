import spacy
import os
import pickle
import torch
import time
from torch import cuda
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

from Task_1_polish.data_processing import Preprocess
from Task_1_polish.get_data import get_batcher, model, Config, weights_init, EarlyStopping

data_path = r"C:\Users\Andrey\PycharmProjects\DeepPavlov"
# preprocesser = Preprocess(path=r'C:\Users\Andrey\Google Диск\DeepPavlov\text8\text8', limit=5, window_size=5, mode='cbow')
# X, y, word2index, index2word = preprocesser._get_data()
# with open(os.path.join(data_path,'intermediate_results', 'w2v_data.pkl'), "wb") as f:
#    pickle.dump([X, y, word2index, index2word], f)

with open(os.path.join(data_path,'intermediate_results', 'w2v_data.pkl'), "rb") as f:
   [X, y, word2index, index2word] = pickle.load(f)

if cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

net = model(len(word2index.keys()), 300, word2index)
net.to(device)

config = Config(lr=0.002, batch_size=2048, num_epochs=100, print_every=100)
optimizer = optim.Adam(net.parameters(), lr=config.lr)
batcher_train = get_batcher(X, y, config.batch_size, device)

# cбросить веса
# net.apply(weights_init)

early_stopping = EarlyStopping()

train_loss = []
train_accuracy = []

start_time = time.time()

for epoch in range(config.num_epochs):

  iter_loss = 0.0
  correct = 0
  iterations = 0

  net.train()  # Put the network into training mode

  for i, (items, classes) in enumerate(batcher_train):
    # Convert torch tensor to Variable
    items = Variable(items)
    classes = Variable(classes)

    net.zero_grad()  # Clear off the gradients from any past operation
    outputs = net(items)  # Do the forward pass

    # config.lr = config.lr / (2 ** (epoch // 10))
    # for param_group in optimizer.param_groups:
    #   param_group['lr'] = config.lr

    loss = F.nll_loss(outputs, classes)
    iter_loss += loss.item()  # Accumulate the loss
    loss.backward()  # Calculate the gradients with help of back propagation
    optimizer.step()  # Ask the optimizer to adjust the parameters based on the gradients

    # Record the correct predictions for training data
    _, predicted = torch.max(outputs.data, 1)

    correct += (predicted == classes.data).sum()
    iterations += 1

    # Record the training loss
    train_loss.append(iter_loss / iterations)
    # Record the training accuracy
    train_accuracy.append(100 * correct / len(batcher_train.dataset))

    if iterations%config.print_every == 0:
       early_stopping.update_loss(train_loss[-1])
       if early_stopping.stop_training():
          break
       print('Epoch %d/%d, Iteration %d, Tr Loss: %.4f'
           % (epoch + 1, config.num_epochs, iterations, train_loss[-1]))

print("--- %s seconds ---" % (time.time() - start_time))

torch.save(net, os.path.join(data_path, 'intermediate_results', 'net.pb'))
