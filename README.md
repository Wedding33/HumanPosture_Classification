# Humanposture Classification

# human-pose-series-classification

- 3 Types of models of <a href="https://github.com/Wedding33/Classification-of-human-posture">human-pose-series-classification</a>.

## Install

- Address:     https://pan.baidu.com/s/1WACOLaTOQqR3s-DEjAgdCw Password:    vzxh
- CNN data: `data_for_CNN_and_Transformer.rar`
- CNN model: `CNNmodel.pth`
- RNN(lstm) data: `RNN_data.zip`
- RNN(lstm) model: `model_1000EPOch_0.77test.pkl`
- Transformer data: `data_for_CNN_and_Transformer.rar`
- Transformer model: `./checkpoint/model.pkl`

## Usage

### CNN train:

- `CNN3.py` is the training code

### CNN test:

- `CNNfinals.py` uses it directly to test

### RNN train:

- Using the `data` in the root of the code, you can use the jupyter notebook to step up and implement 
- (and you can see the `previous training records` in jupyter directly, and I've trained to test accuracy= 0.8).

### RNN test:

- In the last three modules of `RNN(LSTM)Train_Test.ipynb`, we can directly load the model `model_1000EPOch_0.77test.pkl` to test the results by making sure that the model of LSTM is defined and the test data set and its initialization (initial and detach) are loaded

### Transformer train:

- Place your dataset in the file `data`, split them into train/test data and place them in the subfolder `train` and `test`.
- Run `main.py`, then the model will be saved in folder `checkpoints` as `model.pkl`.

### Transformer test:

- Place the pretrained model in folder `checkpoints` and name it as `model.pkl`.
- Open `main.py` and comment line 178

  ```python
  path = train(device, train_loader=train_loader, valid_loader=valid_loader, epochs=epochs)
  ```

- Run `main.py`, then it will show test accuracy with your test data.
