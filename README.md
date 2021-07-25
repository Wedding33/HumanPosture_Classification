# Classification-of-human-posture

# human-pose-series-classification

3 Types of models of <a href="https://github.com/Wedding33/Classification-of-human-posture">human-pose-series-classification</a>.

## Install

CNN data:
CNN model:
RNN(lstm) data:
RNN(lstm) model:
Transformer data:
Transformer model:

## Usage

### CNN train:
-

### To train:

- Place your dataset in the file `data`, split them into train/test data and place them in the subfolder `train` and `test`.

- Run `main.py`, then the model will be saved in folder `checkpoints` as `model.pkl`.

### To test:

- Place the pretrained model in folder `checkpoints` and name it as `model.pkl`.

- Open `main.py` and comment line 178

  ```python
  path = train(device, train_loader=train_loader, valid_loader=valid_loader, epochs=epochs)
  ```

- Run `main.py`, then it will show test accuracy with your test data.
