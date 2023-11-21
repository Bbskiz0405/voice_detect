import warnings

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader

from voice_detect.ffnn.dataset import FFNNDataset
from voice_detect.ffnn.model import FFNN

warnings.filterwarnings('ignore', category=UserWarning)

# set basic parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps')


def load_model(model_path: str):
    model = FFNN()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)
    model.eval()
    return model


def get_ffnn_test_loader(data_path: str, batch_size: int) -> torch.utils.data.DataLoader:
    # get testing dataset
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
    test_data = FFNNDataset(torch.FloatTensor(x), torch.FloatTensor(y))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return test_loader


def evaluate(model_path: str, eval_data_dir: str, batch_size: int = 32):
    # Set parameters
    pred_list = torch.zeros(0, dtype=torch.long, device='cpu')
    ground_truth = torch.zeros(0, dtype=torch.long, device='cpu')

    # Load trainned model
    model = load_model(model_path)

    # Load test data
    test_loader = get_ffnn_test_loader(eval_data_dir, batch_size)

    # start testing
    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs.data, 1)

            pred_list = torch.cat([pred_list, preds.view(-1).cpu()])
            ground_truth = torch.cat([ground_truth, y.view(-1).cpu()])

    # accuracy score
    print('\nAccuracy Score:')
    print(accuracy_score(ground_truth.numpy(), pred_list.numpy()))

    # confusion matrix
    print('\nConfusion Matrix:')
    conf_mat = confusion_matrix(ground_truth.numpy(), pred_list.numpy())
    print(conf_mat)

    # per-class accuracy
    print('\nPer-Class Accuracy:')
    print(100 * conf_mat.diagonal() / conf_mat.sum(1))

    # classification report
    print('\nClassification Report:')
    print(classification_report(ground_truth.numpy(), pred_list.numpy()))
