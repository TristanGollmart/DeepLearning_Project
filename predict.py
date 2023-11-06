import json
import time

from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor, ConvertImageDtype
from tqdm import tqdm

from data_utils.data_stats import MEAN_DICT, STD_DICT
from models import get_architecture
import torch
import torchvision
import torchvision.transforms as transforms

from utils.metrics import AverageMeter, topk_acc, real_acc


@torch.no_grad()
def test(model, loader, loss_fn, args):
    start = time.time()
    model.eval()
    total_acc, total_top5, total_loss = AverageMeter(), AverageMeter(), AverageMeter()

    for ims, targs in tqdm(loader, desc="Evaluation"):
        ims = torch.reshape(ims, (ims.shape[0], -1))
        preds = model(ims)

        if args['dataset'] != 'imagenet_real':
            acc, top5 = topk_acc(preds, targs, k=5, avg=True)
            loss = loss_fn(preds, targs).item()
        else:
            acc = real_acc(preds, targs, k=5, avg=True)
            top5 = 0
            loss = 0

        total_acc.update(acc, ims.shape[0])
        total_top5.update(top5, ims.shape[0])
        total_loss.update(loss)

    end = time.time()

    return (
        total_acc.get_avg(percentage=True),
        total_top5.get_avg(percentage=True),
        total_loss.get_avg(percentage=False),
        end - start,
    )


def predict_cifar10(checkpoint_path, config_path, device='cpu'):
    mean = MEAN_DICT['cifar10']
    std = STD_DICT['cifar10']
    transform = transforms.Compose(
        [
            ToTensor(),
            # ToTorchImage(),
            ConvertImageDtype(torch.float32),
            torchvision.transforms.Resize(size=(64, 64)),
            torchvision.transforms.Normalize(mean, std),
        ]
    )

    batch_size = 4

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ############ load model
    # Path to pre-trained checkpoints and the corresponding config file (e.g. the checkpoints downloaded from given link)

    # model, architecture, crop_resolution, norm = model_from_config(checkpoint_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = get_architecture(**config)
    print('Loading checkpoint', checkpoint_path)

    params = {
        k: v
        for k, v in torch.load(checkpoint_path, map_location=torch.device(device)).items()
    }
    model.load_state_dict(params, strict=False)
    ##########
    loss_fn = CrossEntropyLoss(label_smoothing=config['smooth'])
    results = test(model, testloader, loss_fn, {'dataset': 'cifar10'})
    print(results)

    # correct = 0
    # total = 0
    # # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         # calculate outputs by running images through the network
    #         outputs = model(images)
    #         # the class with the highest energy is what we choose as prediction
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == '__main__':
    MODEL_CHECKPOINT_PATH = '/Users/roncaglionidaniele/Documents/CAS/Deep_Learning/Project/checkpoints/BottleneckMLP_B_6-Wi_512/epoch_900'
    MODEL_CONFIG_PATH = '/Users/roncaglionidaniele/Documents/CAS/Deep_Learning/Project/checkpoints/BottleneckMLP_B_6-Wi_512/config.txt'
    predict_cifar10(MODEL_CHECKPOINT_PATH, MODEL_CONFIG_PATH)
