import os
import torch
import random
import numpy as np
from glob import glob
from torch import nn
import torch.optim as optim
from tqdm.notebook import tqdm
from model import TransformerModel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class MovementDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __transform(self,file_path):
        data = np.load(file_path)
        data[:, 1, :, :, :] = 0
        flag = 0 if np.all(data[:, :, :, :, 1] == 0) else 1
        if flag == 0:

            # 丢弃一层位置信息
            data = np.concatenate((data[:, 0, :, :, 0], data[:, 2, :, :, 0]))
            data_transformed = data[:, 0, :].copy()
            data_transformed.resize(1, 34)
            for i in range(1, data.shape[1]):
                data_add = data[:, i, :].copy()
                data_add.resize(1, 34)
                data_transformed = np.concatenate((data_transformed, data_add))
                data_transformed = data_transformed.copy() # size = 帧数*34
            
            # 将数据通过插值扩展到301帧
            data_column = data_transformed[:, 0].copy()
            x = np.arange(data_column.shape[0])
            xnew = np.linspace(x[0], x[-1], 301)
            data_interp = np.interp(xnew, x, data_column)
            data_interp.resize(data_interp.shape[0], 1)
            for i in range(1, data_transformed.shape[1]):
                data_column = data_transformed[:, i].copy()
                data_column_interp = np.interp(xnew, x, data_column)
                data_column_interp.resize(data_column_interp.shape[0], 1)
                data_interp = np.concatenate((data_interp, data_column_interp), axis=1)
        return torch.from_numpy(data_interp).float()

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        data_path = self.file_list[idx]
        data_transformed = self.__transform(data_path)

        label = int(data_path.split('\\')[3])

        return data_transformed, label



def train(device, train_loader, valid_loader, epochs):
    model = TransformerModel(flag=0).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)


    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        i = 0
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            print('\rtrain data {}/{}, train loss = {:4f}'.format(i, len(train_loader), epoch_loss), flush=True, end='')
            i += 1
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            i = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)
                print('\rvalid data {}/{}, valid loss = {:4f}'.format(i, len(valid_loader), epoch_loss), flush=True, end='')
                i += 1

        print(
            f"\nEpoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
    torch.save(model, './checkpoints/model.pkl')
    path = './checkpoints/model.pkl'
    return path

def test(path, test_loader, device):
    model = torch.load(path)
    model.eval()
    accuracy = 0
    criterion = nn.CrossEntropyLoss()
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        acc = (output.argmax(dim=1) == label).float().mean()
        accuracy += acc / len(test_loader)

    print('test accuracy = {:4f}'.format(accuracy))

if __name__ == '__main__':
    # Training settings
    batch_size = 1
    epochs = 10
    lr = 3e-6
    gamma = 0.7
    seed = 42
    device = 'cuda'
    path = './checkpoints/model.pkl'
    seed_everything(seed)

    train_dir = '.\\data\\train'
    test_dir = '.\\data\\test'

    train_list = glob(train_dir+'\\*\\*.npy')
    test_list = glob(test_dir+'\\*\\*.npy')

    labels = [int(path.split('\\')[3]) for path in train_list]

    train_list, valid_list = train_test_split(train_list, 
                                            test_size=0.2,
                                            stratify=labels,
                                            random_state=seed)

    print('Train Data: {0}'.format(len(train_list)))
    print('Validation Data: {0}'.format(len(valid_list)))
    print('Test Data: {0}'.format(len(test_list)))
    train_data = MovementDataset(train_list)
    valid_data = MovementDataset(valid_list)
    test_data = MovementDataset(test_list)

    train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

    path = train(device, train_loader=train_loader, valid_loader=valid_loader, epochs=epochs)
    test(path, test_loader=test_loader, device=device)