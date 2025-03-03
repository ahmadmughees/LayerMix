# adapted from https://github.com/andyzoujm/pixmix/blob/main/cifar.py
import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from data import MixerDataset
from models.ResNeXt_DenseNet.densenet import densenet
from models.ResNeXt_DenseNet.resnext import resnext29
from models.ResNet.resnet import resnet18
from models.WideResNet.wideresnet import WideResNet
from transforms import augs_image, augs_spatial, add, multiply, random_pixels, random_elems
from utils.adversarial import PGD
from utils.calibration import calib_err, aurra
from utils.corruptions import test_c, test
from utils.perturbation import test_p

parser = argparse.ArgumentParser(description="Trains a CIFAR Classifier",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"],
                    help="Choose between CIFAR-10, CIFAR-100.")
parser.add_argument("--data-path", type=str, default="./data", help="Path to CIFAR and CIFAR-C directories")
parser.add_argument("--mixing-set", type=str, default="./data/fractals", help="Pixels Mixing set directory.")
parser.add_argument("--model", type=str, default="wrn", choices=["wrn", "resnext", "resnet", "densenet"],
                    help="Choose architecture from wrn, resnext, resnet")
# training set
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
parser.add_argument("--learning-rate", "-lr", type=float, default=0.1, help="Initial learning rate.")
parser.add_argument("--batch-size", "-b", type=int, default=128, help="Batch size.")
parser.add_argument("--eval-batch-size", type=int, default=128)
# optimization options
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
parser.add_argument("--decay", "-wd", type=float, default=0.0005, help="Weight decay (L2 penalty).")
# wrn options
parser.add_argument("--layers", default=40, type=int, help="total number of layers")
parser.add_argument("--widen-factor", default=4, type=int, help="Widen factor")
# augmentation options
parser.add_argument("--mixing-type", type=str, default="layermix",
                    help="Choose mixing type from ipmix, pixmix, layermix")
parser.add_argument("--depth", default=3, type=int, help="Number of augmentation chains to mix per augmented example")
parser.add_argument("--width", default=3, type=int, help="Number of augmentation for image. ")
parser.add_argument("--jsd", action="store_true", help="Turn on JSD consistency loss.")
parser.add_argument("--blending_ratio", default=3, type=float, help="blending ratio")
parser.add_argument("--magnitude", default=3, type=int, help="Severity of base augmentation operators")
# Checkpointing options
parser.add_argument("--save", type=str, default="./result", help="Folder to save checkpoints.")
parser.add_argument("--resume", type=str, default="", help="Checkpoint path for resume / test.")
parser.add_argument("--evaluate", action="store_true", help="Eval only.")
# Acceleration
parser.add_argument("--num-workers", type=int, default=4, help="Number of pre-fetching threads.")
# seed
parser.add_argument("--seed", type=int, default=1, help="Default seed.")
# grayscale fractals
parser.add_argument("--gray-scale-fractals", dest="gray_scale_fractals", action="store_true",
                    help="use gray scale fractals")

args = parser.parse_args()
print(args)

NUM_CLASSES = 100 if args.dataset == "cifar100" else 10
DEVICE = "cuda"
IMAGE_SIZE = 32

scaler = GradScaler()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    print(f"using {seed} as default seed")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def write_to_json(data: dict[str, Any], file_path: str):
    existing_data = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            existing_data = json.load(f)
    for key, value in data.items():
        if key in existing_data and existing_data[key] != value:
            existing_data[f"{key}_old"] = existing_data[key]
        existing_data[key] = value
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)


def train_one_epoch(
        net: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler
) -> float:
    """Train for one epoch."""
    net.train()
    loss_ema = 0.
    for images, targets in train_loader:
        optimizer.zero_grad(set_to_none=True)

        if not args.jsd:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = net(images)
                loss = F.cross_entropy(logits, targets)
        else:
            images_all = torch.cat(images, 0).cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            with (torch.autocast(device_type="cuda", dtype=torch.float16)):
                logits_all = net(images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))

                # Cross-entropy is only computed on clean images
                loss = F.cross_entropy(logits_clean, targets)
                p_clean = F.softmax(logits_clean, dim=1)
                p_aug1 = F.softmax(logits_aug1, dim=1)
                p_aug2 = F.softmax(logits_aug2, dim=1)

                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                loss += 12 * (F.kl_div(p_mixture, p_clean, reduction="batchmean") +
                              F.kl_div(p_mixture, p_aug1, reduction="batchmean") +
                              F.kl_div(p_mixture, p_aug2, reduction="batchmean")) / 3.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    return loss_ema


def run_full_evaluate(
        net: nn.Module,
        test_loader: DataLoader,
        test_data: Dataset,
        paths: dict,
        num_classes: int,
        seed: int
) -> None:

    evaluate_pgd = True
    evaluate_rms = True
    evaluate_c = True
    evaluate_c_bar = True
    evaluate_p = False

    results = dict()
    results["seed"] = seed

    start = time.time()
    # Evaluate clean accuracy first because test_c mutates underlying data
    net.eval()
    ctx = test(net, test_loader)

    results["test_loss"] = ctx.loss
    results["test_acc"] = ctx.acc
    results["test_error"] = 100 - 100. * ctx.acc

    if evaluate_rms:
        rms = calib_err(ctx.confidence, ctx.correct, p="2")
        aurra_value = aurra(ctx.confidence, ctx.correct)
        results["rms"] = rms
        results["aurra"] = aurra_value

    # test on adversarial data
    if evaluate_pgd:
        adversary = PGD(epsilon=2. / 255, num_steps=20, step_size=0.5 / 255).cuda()
        adv_ctx = test(net, test_loader, adv=adversary)
        results["adv_test_loss"] = adv_ctx.loss
        results["adv_test_acc"] = adv_ctx.acc
        results["adv_test_error"] = 100 - 100. * adv_ctx.acc

    # test on corrupted data
    if evaluate_c:
        test_c_ctx = test_c(
            net, test_data, paths.get("c_path"),
            eval_batch_size=args.eval_batch_size,
            workers=args.num_workers,
            save_path=args.save
        )
        results["c_acc"] = test_c_ctx.acc
        results["c_error"] = 100 - 100. * test_c_ctx.acc
        results["c_rms"] = test_c_ctx.rms

    # test on c bar
    if evaluate_c_bar:
        test_c_bar_ctx = test_c(
            net, test_data, paths.get("c_bar_path"),
            eval_batch_size=args.eval_batch_size,
            workers=args.num_workers,
            save_path=args.save)
        results["c_bar_acc"] = test_c_bar_ctx.acc
        results["c_bar_error"] = 100 - 100. * test_c_bar_ctx.acc
        results["c_bar_rms"] = test_c_bar_ctx.rms
        if evaluate_c:
            results["mce_c_and_c_bar"] = 100 - 100. * (15 * test_c_ctx.acc + 10 * test_c_bar_ctx.acc) / 25

    # test on perturbed data
    if evaluate_p:
        flip_list, top5_list = test_p(net=net, path=paths.get("p_path"), num_classes=num_classes,
                                      save_path=os.path.join(args.save, "perturbation_results.csv"))
        results["mean_flipping_prob"] = np.mean(flip_list)
        results["mean_top5_distance"] = np.mean(top5_list)

    write_to_json(results, os.path.join(args.save, f"all_results.json"))

    print(f"evaluation finished in {time.time() - start / 60:.2f} minutes.")
    print("The results are: ")
    for key, value in results.items():
        print(f"{key} : {value}")
    return


def get_lr(step: int, total_steps: int, lr_max: float, lr_min: float, warmup_steps: int) -> float:
    """Compute learning rate with linear warmup and cosine annealing."""
    if step < warmup_steps:
        return lr_min + (lr_max - lr_min) * (step / warmup_steps)
    else:  # cosine annealing
        return lr_min + (lr_max - lr_min) * 0.5 * (
                    1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))


def main():
    SEED = args.seed if args.seed>0 else random.randint(1, 1000)
    set_seed(SEED)

    normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
    to_tensor_norm = transforms.Compose([transforms.ToTensor(), normalize])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMAGE_SIZE, padding=4)])

    if args.dataset == "cifar10":
        train_data = datasets.CIFAR10(
            args.data_path,
            train=True,
            transform=train_transform,
            download=True
        )
        test_data = datasets.CIFAR10(
            args.data_path,
            train=False,
            transform=to_tensor_norm,
            download=True
        )
        paths = {
            "c_path": os.path.join(args.data_path, "CIFAR-10-C/"),
            "c_bar_path" : os.path.join(args.data_path, "CIFAR-10-C-Bar/"),
            "p_path": os.path.join(args.data_path, "CIFAR-10-P/"),
        }
        num_classes = 10
    else:
        train_data = datasets.CIFAR100(
            args.data_path,
            train=True,
            transform=train_transform,
            download=True)
        test_data = datasets.CIFAR100(
            args.data_path,
            train=False,
            transform=to_tensor_norm,
            download=True)
        paths = {
            "c_path": os.path.join(args.data_path, "CIFAR-100-C/"),
            "c_bar_path": os.path.join(args.data_path, "CIFAR-100-C-Bar/"),
            "p_path": os.path.join(args.data_path, "CIFAR-100-P/"),
        }
        num_classes = 100

    mixing_set_transforms = [
        transforms.Resize(36),
        transforms.RandomCrop(32),
    ]
    if args.gray_scale_fractals:
        print("Using grayscale fractals")
        mixing_set_transforms += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Grayscale(3)
        ]
    else:
        print("Attention!! Using colored fractals")

    mixing_set = datasets.ImageFolder(args.mixing_set, transform=transforms.Compose(mixing_set_transforms))
    print("train_size", len(train_data))
    print("aug_size", len(mixing_set))

    blending_fns = {
        "baseline"  : [],
        "pixmix"    : [add, multiply],
        "ipmix"     : [add, multiply, random_pixels, random_elems],
        "layermix"  : [add, multiply, add, multiply, random_pixels, random_elems],
    }.get(args.mixing_type)

    assert blending_fns is not None, f"blending functions not found for {args.mixing_type}"
    print("using blending functions:", blending_fns)

    train_data = MixerDataset(
        dataset=train_data,
        mixing_set=mixing_set,
        depth=args.depth,
        width=args.width,
        image_aug_fns=augs_image,
        spatial_aug_fns=augs_spatial,
        blending_fns=blending_fns,
        magnitude=args.magnitude,
        blending_ratio=args.blending_ratio,
        mixer_type=args.mixing_type,
        jsd=args.jsd,
        normalize=normalize)

    # Fix dataloader worker issue
    # https://github.com/pytorch/pytorch/issues/5059
    def wif(id):
        uint64_seed = torch.initial_seed()
        ss = np.random.SeedSequence([uint64_seed])
        # More than 128 bits (4 32-bit words) would be overkill.
        np.random.seed(ss.generate_state(4))

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=wif)

    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    # load model
    net = {
        "densenet": densenet(num_classes=num_classes),
        "wrn": WideResNet(args.layers, num_classes, args.widen_factor, drop_rate=0.3),
        "resnext": resnext29(num_classes=num_classes),
        "resnet": resnet18(num_classes=num_classes),
    }.get(args.model)
    assert net is not None, f"model not found for {args.model}"

    net = net.cuda()  # move to cuda

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location="cuda")
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Model restored from epoch:", start_epoch)

    if args.evaluate:
        run_full_evaluate(net, test_loader, test_data, paths, num_classes=NUM_CLASSES, seed=SEED)
        return

    net = torch.compile(net)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate,
            0 * len(train_loader))  # 0 here does not do warm up. change the value for warmup
    )

    if os.path.exists(args.save):
        args.save = str(args.save) + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(args.save, exist_ok=True)

    # save training args to a file
    with open(os.path.join(args.save, "training_args.json"), "a+") as f:
        json.dump({str(datetime.now()): args.__dict__}, f, indent=4)

    log_path = os.path.join(args.save, args.dataset + "_" + args.model + "_training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,time(s),train_loss,test_loss,test_error(%)\n")

    best_acc = 0
    print("Beginning training from epoch:", start_epoch + 1)
    for epoch in range(start_epoch, args.epochs):
        begin_time = time.time()

        train_loss_ema = train_one_epoch(net, train_loader, optimizer, scheduler)
        ctx = test(net, test_loader)
        test_loss = ctx.loss
        test_acc = ctx.acc
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        checkpoint = {
            "epoch": epoch,
            "dataset": args.dataset,
            "model": args.model,
            "state_dict": net._orig_mod.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }

        save_path = os.path.join(args.save, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(args.save, "model_best.pth.tar"))

        with open(log_path, "a") as f:
            f.write(
                f"{int((epoch + 1)):3d}, "
                f"{int(time.time() - begin_time):5d}, "
                f"{train_loss_ema:.6f}, "
                f"{test_loss:.5f}, "
                f"{100 - 100. * test_acc:.2f} "
                f"\n")

        print(f"Epoch {(epoch + 1):3d} | "
              f"Time {int(time.time() - begin_time):5d} | "
              f"Train Loss {train_loss_ema:.4f} | "
              f"Test Loss {test_loss:.3f} | "
              f"Test Error {100 - 100. * test_acc:.2f} | "
              f"LR {optimizer.param_groups[0]['lr']:.8f}")

    # Evaluate the best saved model
    best_path = torch.load(os.path.join(args.save, "model_best.pth.tar"), map_location="cuda")
    print(f"evaluating the best model {best_path['epoch'] + 1}")
    net._orig_mod.load_state_dict(best_path["state_dict"])
    run_full_evaluate(net, test_loader, test_data, paths, num_classes=NUM_CLASSES, seed=SEED)


if __name__ == "__main__":
    main()
