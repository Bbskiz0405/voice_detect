import warnings

import torch
import torch.utils.data
import torchvision
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torchvision.transforms import transforms
#test
from voice_detect.cnn.model import CNN

warnings.filterwarnings('ignore', category=UserWarning)

# set basic parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps')


def load_model(model_path: str):
    model = CNN()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)
    model.eval()
    return model


def get_cnn_test_loader(data_dir: str, batch_size: int) -> torch.utils.data.DataLoader:
    # set basic transforms. Spectrograms have to look a certain way so rotations, flips, and other
    # transforms do not make sense in this application
    transform = {'test': transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()])}

    # get testing dataset
    test_data = torchvision.datasets.ImageFolder(root=data_dir,
                                                 transform=transform['test'])

    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return test_loader


def evaluate(model_path: str, eval_data_dir: str, batch_size: int = 16):
    # Set parameters
    pred_list = torch.zeros(0, dtype=torch.long, device='cpu')
    ground_truth = torch.zeros(0, dtype=torch.long, device='cpu')

    # Load trainned model
    model = load_model(model_path)

    # Load test data
    test_loader = get_cnn_test_loader(eval_data_dir, batch_size)

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
