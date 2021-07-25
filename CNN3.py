import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import glob


device = 'cuda' if torch.cuda.is_available() else 'cpu'

root0 = r'.\data\train\000'
root1 = r'.\data\train\001'
root2 = r'.\data\train\002'
root3 = r'.\data\train\003'
root4 = r'.\data\train\004'
f_train = glob.glob(root0+'//*.NPY')+glob.glob(root1+'//*.NPY')+glob.glob(root2+'//*.NPY')+glob.glob(root3+'//*.NPY')+glob.glob(root4+'//*.NPY')
train_label = np.append(np.append(np.append(np.append(np.zeros(700), np.ones(700)), np.ones(700)*2), np.ones(700)*3), np.ones(610)*4)
train_label = torch.LongTensor(train_label).to(device)


root0 = r'.\data\test\000'
root1 = r'.\data\test\001'
root2 = r'.\data\test\002'
root3 = r'.\data\test\003'
root4 = r'.\data\test\004'
f_test = glob.glob(root0+'//*.NPY')+glob.glob(root1+'//*.NPY')+glob.glob(root2+'//*.NPY')+glob.glob(root3+'//*.NPY')+glob.glob(root4+'//*.NPY')
test_y = np.append(np.append(np.append(np.append(np.zeros(27), np.ones(27)), np.ones(27)*2), np.ones(26)*3), np.ones(10)*4)
test_y = torch.LongTensor(test_y)

n = 0
train_x = np.zeros([3410, 51, 128])
for name in f_train:
    sample = np.load(name)[0, :, :, :, 0]
    train_x[n] = np.resize(sample, (51, 128))
    n = n+1
train_x = torch.LongTensor(train_x)
train_x = torch.unsqueeze(train_x, dim=1).type(torch.LongTensor)
train_x = train_x.to(device)

test_x = np.zeros([117, 51, 128])
n = 0
for name in f_test:
    sample = np.load(name)[0, :, :, :, 0]
    test_x[n] = np.resize(sample, (51, 128))
    n = n+1
test_x = torch.FloatTensor(test_x)
test_x = torch.unsqueeze(test_x, dim=1).type(torch.FloatTensor)
test_x = test_x.to(device)

# Hyper Parameters
EPOCH = 200               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 310
LR = 0.0002              # learning rate


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(1536,100)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.relu = torch.nn.ReLU()
        self.mlp2 = torch.nn.Linear(100,5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.dropout(x)
        # x = self.relu(x)
        output = self.mlp2(x)
        return output, x


cnn = CNN().to(device)
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested


from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = False
except: HAS_SK = False; print('Please install sklearn for layer visualization')

def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()
# training and testing

cnn.train()
for epoch in range(EPOCH):
    for step in range(11):   # gives batch data, normalize x when iterate train_loader
        b_x = train_x[BATCH_SIZE*step:BATCH_SIZE*step+BATCH_SIZE, :, :]
        b_y = train_label[BATCH_SIZE*step:BATCH_SIZE*step+BATCH_SIZE]
        b_x = b_x.float().to(device)
        b_y = b_y.long().to(device)
        # print('b_x', b_x.shape)
        # print('b_y', b_y.shape)
        output = cnn(b_x)[0]               # cnn output
        # print(output.size())
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.to('cpu').numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.to('cpu').numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.to('cpu').numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
plt.ioff()

torch.save(cnn, './model.pth')

cnn.eval()
# print 10 predictions from test data
test_output, _ = cnn(test_x)
pred_y = torch.max(test_output, 1)[1].data.to('cpu').numpy()
print(pred_y, 'prediction number')
print(test_y.numpy(), 'real number')
correct_count = 0
for pred, real in zip(pred_y, test_y):
    if pred == real:
        correct_count += 1
print('Final accuracy: %.2f'%(correct_count/pred_y.shape[0]))
