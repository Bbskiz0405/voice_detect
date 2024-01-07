# built-in modules
import logging
import os
import warnings
from datetime import date

# 3rd party modules
import pandas as pd
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

# self-defined modules
from voice_detect.config import abs_ffnn_model_path
from voice_detect.ffnn.dataset import FFNNDataset
from voice_detect.ffnn.model import FFNN
from voice_detect.utils.gen_utils import get_accuracy

warnings.filterwarnings('ignore', category=UserWarning)


# current date
today = date.today()
today = today.strftime('%Y-%m-%d')

# set basic parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps')


def get_train_loader(data_path: str, batch_size: int):
    # get training set
    df = pd.read_csv(data_path, header=None)

    # encode labels
    df.iloc[:, -1] = df.iloc[:, -1].map({'voice': 1.0, 'not_voice': 0})

    # remove file names
    df.drop(df.columns[0], axis=1, inplace=True)

    # seperate data and labels
    x = df.iloc[:, 0: -1].values
    y = df.iloc[:, -1].values
    y = y.astype('float32')

    # get dataloader for training
    train_data = FFNNDataset(torch.FloatTensor(x), torch.FloatTensor(y))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    return train_loader


def train_ffnn(data_path: str, epochs: int = 250, batch_size: int = 32, lr: int = 0.005):
    # Set up model name
    model_file_name = 'FFNN-' + today + '.pt'

    # Set up logging
    log_file_name = 'FFNN-' + today + '.log'
    logging.basicConfig(filename=os.path.join(abs_ffnn_model_path, log_file_name),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        level=logging.INFO)

    # Get train loader
    train_loader = get_train_loader(data_path, batch_size)

    model = FFNN()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0
        epoch_accuracy = 0
        epoch_steps = 0

        for _, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device, dtype=torch.int64)

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1

            epoch_accuracy += get_accuracy(output, y)

        epoch_accuracy /= epoch_steps

        # print status onto terminal and log file
        print('Epoch: [%d/%d] | Loss: %.3f | Accuracy: %.3f' % (epoch + 1, epochs, epoch_loss, epoch_accuracy))
        logging.info('Epoch: [%d/%d] | Loss: %.3f | Accuracy: %.3f' % (epoch + 1, epochs, epoch_loss, epoch_accuracy))

    # save model
    torch.save(model.state_dict(), os.path.join(abs_ffnn_model_path, model_file_name))
