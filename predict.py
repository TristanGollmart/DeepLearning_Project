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
    # get test data
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
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    # load model from checkpoint
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['num_classes'] = 10

    model = get_architecture(**config)
    print('Loading checkpoint', checkpoint_path)

    params = {
        k: v
        for k, v in torch.load(checkpoint_path, map_location=torch.device(device)).items()
    }
    model.load_state_dict(params, strict=False)
    loss_fn = CrossEntropyLoss(label_smoothing=config['smooth'])
    # run test
    results = test(model, test_loader, loss_fn, {'dataset': 'cifar10'})
    print(results)


if __name__ == '__main__':
    # MODEL_CHECKPOINT_PATH = '/Users/roncaglionidaniele/Documents/CAS/Deep_Learning/Project/checkpoints/BottleneckMLP_B_12-Wi_1024/epoch_900'
    MODEL_CHECKPOINT_PATH = '/Users/roncaglionidaniele/Documents/CAS/Deep_Learning/Project/checkpoints/BottleneckMLP_B_12-Wi_1024/cifar10_epoch_20'
    MODEL_CONFIG_PATH = '/Users/roncaglionidaniele/Documents/CAS/Deep_Learning/Project/checkpoints/BottleneckMLP_B_12-Wi_1024/config.txt'
    predict_cifar10(MODEL_CHECKPOINT_PATH, MODEL_CONFIG_PATH)
