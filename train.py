import argparse
import os
import time
import json

import torch
import wandb
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from models import get_architecture
from data_utils.data_stats import *
from data_utils.dataloader import get_loader
from utils.config import config_to_name
from utils.get_compute import get_compute
from utils.metrics import topk_acc, real_acc, AverageMeter
from utils.optimizer import get_optimizer, get_scheduler, OPTIMIZERS_DICT, SCHEDULERS
from few_shot import get_loaders as get_few_shot_loaders
from few_shot import do_few_shot


def train(model, opt, scheduler, loss_fn, epoch, train_loader, args):
    start = time.time()
    model.train()

    total_acc, total_top5 = AverageMeter(), AverageMeter()
    total_loss = AverageMeter()

    for step, (ims, targs) in enumerate(
        tqdm(train_loader, desc="Training epoch: " + str(epoch))
    ):
        ims = torch.reshape(ims, (ims.shape[0], -1))
        preds = model(ims)

        if args.mixup > 0:
            targs_perm = targs[:, 1].long()
            weight = targs[0, 2].squeeze()
            targs = targs[:, 0].long()
            if weight != -1:
                loss = loss_fn(preds, targs) * weight + loss_fn(preds, targs_perm) * (
                    1 - weight
                )
            else:
                loss = loss_fn(preds, targs)
                targs_perm = None
        else:
            loss = loss_fn(preds, targs)
            targs_perm = None

        acc, top5 = topk_acc(preds, targs, targs_perm, k=5, avg=True)
        total_acc.update(acc, ims.shape[0])
        total_top5.update(top5, ims.shape[0])

        loss.backward()
        if step % args.accum_steps == 0:
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            opt.zero_grad()

        total_loss.update(loss.item(), ims.shape[0])

    end = time.time()

    scheduler.step()

    return (
        total_acc.get_avg(percentage=True),
        total_top5.get_avg(percentage=True),
        total_loss.get_avg(percentage=False),
        end - start,
    )


@torch.no_grad()
def test(model, loader, loss_fn, args):
    start = time.time()
    model.eval()
    total_acc, total_top5, total_loss = AverageMeter(), AverageMeter(), AverageMeter()

    for ims, targs in tqdm(loader, desc="Evaluation"):
        if ims.device != args.device:
            ims = ims.to(args.device)
            targs = targs.to(args.device)

        ims = torch.reshape(ims, (ims.shape[0], -1))
        preds = model(ims)

        if args.dataset != "imagenet_real":
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


def main(args):
    # Use mixed precision matrix multiplication
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    model = get_architecture(**args.__dict__).cuda()

    # Count number of parameters for logging purposes
    args.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create unique identifier
    name = config_to_name(args)
    path = os.path.join(args.checkpoint_folder, name)

    # Create folder to store the checkpoints
    if not os.path.exists(path):
        os.makedirs(path)
        with open(path + "/config.txt", "w") as f:
            json.dump(args.__dict__, f, indent=2)

    # Get the dataloaders
    local_batch_size = args.batch_size // args.accum_steps

    train_loader = get_loader(
        args.dataset,
        bs=local_batch_size,
        mode="train",
        augment=args.augment,
        dev=device,
        num_samples=args.n_train,
        mixup=args.mixup,
        data_path=args.data_path,
        data_resolution=args.resolution,
        crop_resolution=args.crop_resolution,
    )

    test_loader = get_loader(
        args.dataset,
        bs=local_batch_size,
        mode="test",
        augment=False,
        dev=device,
        data_path=args.data_path,
        data_resolution=args.resolution,
        crop_resolution=args.crop_resolution,
    )

    if args.downstream_dataset is not None:
        downstream_train_loader, downstream_test_loader = get_few_shot_loaders(
            args.downstream_dataset,
            args.downstream_data_resolution,
            args.downstream_n_few_shot,
            args.downstream_num_classes,
            args.__dict__,
        )

    start_ep = 1
    if args.reload:
        try:
            params = torch.load(path + "/name_of_checkpoint")
            model.load_state_dict(params)
            start_ep = 350
        except:
            print("No pretrained model found, training from scratch")

    opt = get_optimizer(args.optimizer)(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_scheduler(opt, args.scheduler, **args.__dict__)

    loss_fn = CrossEntropyLoss(label_smoothing=args.smooth)

    if args.wandb:
        # Add your wandb credentials and project name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args.__dict__,
            tags=["pretrain", args.dataset],
        )
        wandb.run.name = name

    compute_per_epoch = get_compute(model, args.n_train, args.crop_resolution)

    for ep in range(start_ep, args.epochs):
        calc_stats = (ep + 1) % args.calculate_stats == 0

        current_compute = compute_per_epoch * ep

        train_acc, train_top5, train_loss, train_time = train(
            model, opt, scheduler, loss_fn, ep, train_loader, args
        )

        if args.wandb:
            wandb.log({"Training time": train_time, "Training loss": train_loss})

        if ep % args.save_freq == 0 and args.save:
            torch.save(
                model.state_dict(),
                path + "/epoch_" + str(ep) + "_compute_" + str(current_compute),
            )

        if calc_stats:
            test_acc, test_top5, test_loss, test_time = test(
                model, test_loader, loss_fn, args
            )

            if args.wandb:
                wandb.log(
                    {
                        "Training accuracy": train_acc,
                        "Training Top 5 accuracy": train_top5,
                        "Test accuracy": test_acc,
                        "Test Top 5 accuracy": test_top5,
                        "Test loss": test_loss,
                        "Inference time": test_time,
                    },
                    step=ep,
                )

            # Print all the stats
            print("Epoch", ep, "       Time:", train_time)
            print("-------------- Training ----------------")
            print("Average Training Loss:       ", "{:.6f}".format(train_loss))
            print("Average Training Accuracy:   ", "{:.4f}".format(train_acc))
            print("Top 5 Training Accuracy:     ", "{:.4f}".format(train_top5))
            print("---------------- Test ------------------")
            print("Current Optimal Accuracy     ", "{:.4f}".format(optimal_acc))
            print("Test Accuracy        ", "{:.4f}".format(test_acc))
            print("Top 5 Test Accuracy          ", "{:.4f}".format(test_top5))
            print()

            if args.downstream_dataset is not None:
                downstream_accuracies = do_few_shot(
                    model,
                    downstream_train_loader,
                    downstream_test_loader,
                    args.downstream_regularization,
                )

                if args.wandb:
                    wandb.log(
                        {
                            "Downstream top1": downstream_accuracies[0],
                            "Downstream top5": downstream_accuracies[1],
                        },
                        step=ep,
                    )

                print(
                    "Downstream accuracies (top1, top5)",
                    downstream_accuracies,
                )


def get_parser():
    parser = argparse.ArgumentParser(description="Scaling MLPs")

    # Data
    parser.add_argument(
        "--data_path", default="./beton", type=str, help="Path to data directory"
    )
    parser.add_argument("--dataset", default="imagenet21", type=str, help="Dataset")
    parser.add_argument("--resolution", default=64, type=int, help="Image Resolution")
    parser.add_argument(
        "--crop_resolution", default=None, type=int, help="Crop Resolution"
    )
    parser.add_argument(
        "--n_train", default=None, type=int, help="Number of samples. None for all"
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to augment data",
    )
    parser.add_argument("--mixup", default=0.8, type=float, help="Strength of mixup")

    # Model
    parser.add_argument(
        "--model", default="BottleneckMLP", type=str, help="Type of model"
    )
    parser.add_argument(
        "--architecture", default="B_6-Wi_1024", type=str, help="Architecture type"
    )
    parser.add_argument(
        "--normalization", default="layer", type=str, help="Normalization type"
    )
    # Training
    parser.add_argument(
        "--optimizer",
        default="lion",
        type=str,
        help="Choice of optimizer",
        choices=OPTIMIZERS_DICT.keys(),
    )
    parser.add_argument("--batch_size", default=4096, type=int, help="Batch size")
    parser.add_argument(
        "--accum_steps", default=1, type=int, help="Number of accumulation steps"
    )
    parser.add_argument("--lr", default=0.00005, type=float, help="Learning rate")
    parser.add_argument(
        "--scheduler", type=str, default="none", choices=SCHEDULERS, help="Scheduler"
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay")
    parser.add_argument("--epochs", default=500, type=int, help="Epochs")
    parser.add_argument(
        "--smooth", default=0.3, type=float, help="Amount of label smoothing"
    )
    parser.add_argument("--clip", default=0.0, type=float, help="Gradient clipping")
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reinitialize from checkpoint",
    )
    # Downstream
    parser.add_argument(
        "--downstream_dataset", default="imagenet", type=str, help="Downstream dataset"
    )
    parser.add_argument(
        "--downstream_data_resolution",
        default=64,
        type=int,
        help="Downstream data resolution",
    )
    parser.add_argument(
        "--downstream_n_few_shot", default=10, type=int, help="Downstream n-shot"
    )
    parser.add_argument("--downstream_regularization", type=float, default=1e-2)
    # Logging
    parser.add_argument(
        "--calculate_stats",
        type=int,
        default=1,
        help="Frequence of calculating stats",
    )
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default="./checkpoints",
        help="Path to checkpoint directory",
    )
    parser.add_argument("--save_freq", type=int, default=50, help="Save frequency")
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save checkpoints",
    )
    parser.add_argument(
        "--wandb",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to log with wandb",
    )
    parser.add_argument(
        "--wandb_project", default="mlps", type=str, help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_entity", default=None, type=str, help="Wandb entity name"
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    args.num_classes = CLASS_DICT[args.dataset]

    if args.downstream_dataset is not None:
        args.downstream_num_classes = CLASS_DICT[args.downstream_dataset]

    if args.n_train is None:
        args.n_train = SAMPLE_DICT[args.dataset]

    if args.crop_resolution is None:
        args.crop_resolution = args.resolution
    if args.wandb_entity is None:
        print("No wandb entity provided, Continuing without wandb")
        args.wandb = False

    main(args)
