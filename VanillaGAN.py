#!/usr/bin/env python
# coding: utf-8

# In[128]:


import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.gridspec as gridspec

start = time.time()

root = './MNIST_data/'
if not os.path.exists(root):
    os.mkdir(root)
trans = transforms.Compose([transforms.ToTensor()])
# trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
print( "torch.device is", device)

train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)
batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)
print("==>>> total trainning batch number: {}".format(len(train_loader)))
print("==>>> total testing batch number: {}".format(len(test_loader)))
print("==>>> total number of batches are: {}".format(batch_size))

for index, batch in enumerate(train_loader):
    inputs = batch[0]
    labels = batch[1]
    if(index == 0):
        print("==>>> input shape of a batch is: {}".format(inputs.shape))
        print("==>>> labels shape of a batch is: {}".format(labels.shape))
        print(inputs[0].shape, labels[0].shape)
        


# In[129]:


plt.gray()
inputs = inputs.numpy()
labels = labels.numpy()
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_classes = len(classes)
samples_per_class = 5
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(labels == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=True)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        im = inputs[idx].reshape(28,28)
        plt.imshow((im*255).astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
    


# In[139]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# Hyperparams
minibatch_size = 100;
z_dim = 100
X_dim = 784
h_dim = 128
y_dim = 1
lr = 1e-3
num_epochs = 100000
k = 1 
c = 0

########## DISCRIMINATOR ##############
D = nn.Sequential(
    nn.Linear(X_dim, h_dim),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(h_dim, y_dim),
    nn.Sigmoid()
    
).to(device)

########### GENERATOR #############
G = nn.Sequential(
    nn.Linear(z_dim, h_dim),
    nn.ReLU(),
    nn.Linear(h_dim, X_dim),
    nn.Sigmoid()
).to(device)

def xavier_init(size):
    inp_dim = size[0]
    xavier_stddev = 1. / np.sqrt(inp_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

def reset_grad():
    G.zero_grad()
    D.zero_grad()

G_lossfn = optim.Adam(G.parameters(), lr)
D_lossfn = optim.Adam(D.parameters(), lr)

ones_label = Variable(torch.ones(minibatch_size, 1)).to(device)
zeros_label = Variable(torch.zeros(minibatch_size, 1)).to(device)

for epoch in range(num_epochs):
    z = Variable(xavier_init(size=[minibatch_size, z_dim]), requires_grad =True).to(device)
#     print("shape of z is: {}".format(z.shape))
    X, labels = mnist.train.next_batch(minibatch_size)
    X=X.reshape(100,784)
    X = Variable(torch.from_numpy(X), requires_grad=True).to(device)
#     print("shape of X is: {}".format(X.shape))
    labels = Variable(torch.from_numpy(labels)).to(device)
    G_sample = G(z).to(device)
#     print("shape of generated sample",G_sample.shape)
    D_real = D(X).to(device)   
#     print("shape of real sample",D_real.shape)
    D_fake = D(G_sample).to(device)
    loss = nn.BCELoss()
    D_loss_real = loss(D_real, ones_label)
    D_loss_fake = loss(D_fake, zeros_label)
    D_loss = D_loss_real + D_loss_fake
    D_loss.backward()
    D_lossfn.step()
    reset_grad()

    z = Variable(xavier_init(size= [minibatch_size, z_dim]), requires_grad = True).to(device)
    G_sample = G(z).to(device)
    D_fake = D(G_sample).to(device)
    G_loss = loss(D_fake, ones_label)
    G_loss.backward()
    G_lossfn.step()
    reset_grad()
        
#     print('Iter-{}; D_loss: {}; G_loss: {}'.format(epoch, D_loss.cpu().data.numpy(), G_loss.cpu().data.numpy()))
    if epoch % 1000 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'.format(epoch, D_loss.cpu().data.numpy(), G_loss.cpu().data.numpy()))
        plt.scatter(epoch, D_loss.cpu().data.numpy())
        plt.scatter(epoch, G_loss.cpu().data.numpy())
        samples = G(z).cpu().data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('GAN_output/'):
            os.makedirs('GAN_output/')

        plt.savefig('GAN_output/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        c += 1
        plt.close(fig)
end = time.time()
print("total time taken for training: {}".format(end - start))
plt.show()


# In[146]:


state = {
    'epoch': epoch,
    'state_dict1': D.state_dict(),
    'state_dict2': G.state_dict(),
    'optimizer1': G_lossfn.state_dict(),
    'optimizer2': D_lossfn.state_dict(),
}
file_name = "checkpoint.pth"
torch.save(state, file_name)


# In[152]:


state_dict = torch.load('checkpoint.pth')
print(state_dict['state_dict1'])


# In[ ]:




