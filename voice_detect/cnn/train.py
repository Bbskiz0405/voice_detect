# built-in modules
import logging
import os
import warnings
from datetime import date

# 3rd party modules
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
from torchvision.transforms import transforms

# self-defined modules
from voice_detect.cnn.model import CNN
from voice_detect.config import abs_cnn_model_path
from voice_detect.utils.gen_utils import get_accuracy

warnings.filterwarnings('ignore', category=UserWarning)

# current date
today = date.today()
today = today.strftime('%Y-%m-%d')

# set basic parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps')


def get_cnn_train_loader(data_dir: str, batch_size: int, workers: int):
    # set basic transforms. Spectrograms have to look a certain way so rotations,
    # flips, and other
    # transforms do not make sense in this application
    transform = {'train': transforms.Compose([transforms.Resize([32, 32]),
                                              transforms.ToTensor()])}

    # get train dataset
    train_data = torchvision.datasets.ImageFolder(root=data_dir,
                                                  transform=transform['train'])

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers)
    return train_loader


def train_cnn(data_dir: str, epochs: int = 250, batch_size: int = 16, lr: int = 0.001, workers: int = 0):
    # set up model name
    model_file_name = 'CNN-' + today + '.pt'

    # set up logging
    log_file_name = 'CNN-' + today + '.log'
    logging.basicConfig(filename=os.path.join(abs_cnn_model_path, log_file_name),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        level=logging.INFO)

    # Get train loader
    train_loader = get_cnn_train_loader(data_dir, batch_size, workers)

    # Initialize model, activate func, and optimizer
    model = CNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0
        epoch_accuracy = 0
        epoch_steps = 0

        for _, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)

            output = model(img)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1

            epoch_accuracy += get_accuracy(output, label)

        epoch_accuracy /= epoch_steps

        # print status onto terminal and log file
        print('Epoch: [%d/%d] | Loss: %.3f | Accuracy: %.3f' % (epoch + 1, epochs, epoch_loss, epoch_accuracy))
        logging.info('Epoch: [%d/%d] | Loss: %.3f | Accuracy: %.3f' % (epoch + 1, epochs, epoch_loss, epoch_accuracy))

    # save model
    torch.save(model.state_dict(), os.path.join(abs_cnn_model_path, model_file_name))
