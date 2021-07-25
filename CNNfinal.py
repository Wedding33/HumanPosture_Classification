import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


cnn = torch.load('./model.pth')
root0 = r'.\data\test\000'
root1 = r'.\data\test\001'
root2 = r'.\data\test\002'
root3 = r'.\data\test\003'
root4 = r'.\data\test\004'
f_test = glob.glob(root0+'//*.NPY')+glob.glob(root1+'//*.NPY')+glob.glob(root2+'//*.NPY')+glob.glob(root3+'//*.NPY')+glob.glob(root4+'//*.NPY')
test_y = np.append(np.append(np.append(np.append(np.zeros(27), np.ones(27)), np.ones(27)*2), np.ones(26)*3), np.ones(10)*4)
test_y = torch.LongTensor(test_y)

test_x = np.zeros([117, 51, 128])
n = 0
for name in f_test:
    sample = np.load(name)[0, :, :, :, 0]
    test_x[n] = np.resize(sample, (51, 128))
    n = n+1
test_x = torch.FloatTensor(test_x)
test_x = torch.unsqueeze(test_x, dim=1).type(torch.FloatTensor)
test_x = test_x.to(device)

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
