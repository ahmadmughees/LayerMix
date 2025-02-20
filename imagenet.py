# adapted from https://github.com/andyzoujm/pixmix/blob/main/imagenet.py
import argparse
import json
import os
import random
import shutil
import tempfile
import time
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms
from PIL import ImageFile
from torch import GradScaler
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from data import MixerDataset
from imagenet_utils import imagenet_r_wnids, imagenet_a_wnids, all_wnids
from transforms import augs_image, augs_spatial, add, multiply, random_pixels, random_elems
from utils.calibration import calib_err
from utils.perturbation import test_imagenet_p

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SIZE = 224

parser = argparse.ArgumentParser(description="ImageNet Training")
parser.add_argument("--data-standard", help="Path to dataset", default="/media/ifiveo/fast/datasets/imagenet-1k/ILSVRC/Data/CLS-LOC/train/")
parser.add_argument("--data-val", help="Path to validation dataset", default="/media/ifiveo/fast/datasets/imagenet-1k/ILSVRC/Data/CLS-LOC/val/")
parser.add_argument("--imagenet-r-dir", default="/media/ifiveo/fast/datasets/imagenet_r", help="Path to ImageNet-R",)
parser.add_argument("--imagenet-c-dir", default="/media/ifiveo/fast/datasets/imagenet_c", help="Path to ImageNet-C",)
parser.add_argument("--imagenet-c-bar-dir", default="/media/ifiveo/fast/datasets/imagenet_c_bar", help="Path to ImageNet-C-Bar",)
parser.add_argument("--imagenet-p-dir", default="/media/ifiveo/fast/datasets/imagenet_p", help="Path to ImageNet-P",)
parser.add_argument("--imagenet-a-dir", default="data/imagenet_a/", help="Path to ImageNet-A",)
parser.add_argument("--mixing-set", default="/media/ifiveo/fast/datasets/deviantart_1", help="Path to mixing set")
parser.add_argument("--num-classes", choices=["200", "1000"], required=True)
parser.add_argument("--magnitude", default=3, type=int, help="Severity of base augmentation operators")
parser.add_argument("--blending_ratio", default=3, type=int, help="Severity of mixing")
# mixing options
parser.add_argument("--depth", default=3, type=int, help="Number of augmentation chains to mix per augmented example,chosen from P and IMG ") #depth is number of branches
parser.add_argument("--width", default=3, type=int, help="Number of augmentation for image. ")  # number of augs in a branch
parser.add_argument("--jsd", action="store_true", help="Turn on JSD consistency loss.")
parser.add_argument("--mixing-type", type=str, default="fractal_mix", help="Choose mixing type from ipmix, pixmix, fractalmix")

parser.add_argument("--save", default="checkpoints/TEMP", type=str)
parser.add_argument("--model", choices=["resnet50", "resnet18"], default="resnet50")
parser.add_argument("--workers", default=8, type=int, metavar="N",
                    help="number of data loading workers (default: 4)")
parser.add_argument("--epochs", default=90, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="N",
                    help="mini-batch size (default: 256), this is the total batch size of all GPUs")
parser.add_argument("--batch-size-val", default=32, type=int)
parser.add_argument("--lr", "--learning-rate", default=0.01, type=float,
                    metavar="LR", help="initial learning rate", dest="lr")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("--wd", "--weight-decay", default=5e-4, type=float,
                    metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
parser.add_argument("-p", "--print-freq", default=10, type=int,
                    metavar="N", help="print frequency (default: 10)")
parser.add_argument("--resume", default="", type=str, metavar="PATH",
                    help="path to latest checkpoint (default: none)")
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true",
                    help="evaluate model on validation set and Imagenet-C")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="use pre-trained model")
parser.add_argument("--gray-scale-fractals", dest="gray_scale_fractals", action="store_true",
                    help="use gray scale fractals")


args = parser.parse_args()
print(args)

# -----------------------------------------------------
# ImageNet-1K classes
classes_chosen_1000 = all_wnids
assert len(classes_chosen_1000) == 1000

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ImageNet-R classes
imagenet_r_wnids.sort()
classes_chosen_200 = imagenet_r_wnids[:]  # Choose 100 classes for our dataset
assert len(classes_chosen_200) == 200
imagenet_r_mask = [wnid in classes_chosen_200 for wnid in all_wnids]

# ImageNet-A classes
imagenet_a_wnids.sort()
assert len(imagenet_a_wnids) == 200
imagenet_a_mask = [wnid in imagenet_a_wnids for wnid in all_wnids]

classes_chosen = classes_chosen_200 if args.num_classes == "200" else classes_chosen_1000

# -----------------------------------------------------
if os.path.exists(args.save):
    resp = "None"
    while resp.lower() not in {"y", "n"}:
        resp = input(f"Save directory {args.save} exits. Continue? [Y/n]: ")
        if resp.lower() == "y":
            break
        elif resp.lower() == "n":
            exit(1)
        else:
            pass
else:
    os.makedirs(args.save)
    print("Created save directory", args.save)


with open(os.path.join(args.save, "training_args.json"), "a+") as f:
    json.dump({str(datetime.now()): args.__dict__}, f, indent=4)


class ImageNetSubsetDataset(datasets.ImageFolder):
    def __init__(self, root, *args, **kwargs):
        self.new_root = tempfile.mkdtemp()
        for _class in classes_chosen:
            orig_dir = os.path.join(root, _class)
            assert os.path.isdir(orig_dir), f"Directory {orig_dir} does not exist"

            os.symlink(orig_dir, os.path.join(self.new_root, _class))
        super().__init__(self.new_root, *args, **kwargs)

    def __del__(self):
        shutil.rmtree(self.new_root)


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


normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
best_acc1 = 0


def main():
    global args, best_acc1
    SEED = 1
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # If using multi-GPU
    cudnn.benchmark = True
    cudnn.deterministic = False

    print(f"=> creating model {args.model}")
    if args.model == "resnet18":
        model = models.resnet18(pretrained=args.pretrained)
        if len(classes_chosen) != 1000:
            model.fc = torch.nn.Linear(512, len(classes_chosen))
    elif args.model == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        if len(classes_chosen) != 1000:
            model.fc = torch.nn.Linear(2048, len(classes_chosen))
    else:
        raise NotImplementedError()
    print(f"=> model created!!")

    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum     = args.momentum,
        weight_decay = args.weight_decay,
        nesterov     = True)

    # optionally resume from a checkpoint
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location="cuda")
            args.start_epoch = checkpoint["epoch"] + 1
            best_acc1 = checkpoint["best_acc1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Model restored from epoch:", args.start_epoch)
        else:
            print(f"=> no checkpoint found at {args.resume}")
    model = torch.compile(model)

    val_loader = DataLoader(
        ImageNetSubsetDataset(
            args.data_val,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=args.batch_size_val, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader_imagenet_r = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.imagenet_r_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size_val, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        evaluate_acc    = False
        evaluate_r      = False
        evaluate_c      = False
        evaluate_c_bar  = False
        evaluate_p      = True
        all_results = dict()
        if evaluate_acc:
            _, val_top1, val_top5, rms = validate(model, val_loader, criterion, args)
            all_results["val_top1"] = val_top1.item()
            all_results["val_top5"] = val_top5.item()
            all_results["val_rms"] = rms
        if evaluate_r:
            _, r_top1, r_top5, rms = validate(model, val_loader_imagenet_r, criterion, args, r=True)
            all_results["r_top1"] = r_top1.item()
            all_results["r_top5"] = r_top5.item()
            all_results["r_rms"] = rms
        if evaluate_acc and evaluate_r:
            with open(os.path.join(args.save, f"eval_results.csv"), "a") as f:
                f.write("val_top1,val_top5,r_top1,r_top5\n")
                f.write(f"{val_top1:0.5f},{val_top5:0.5f},{r_top1:0.5f},{r_top5:0.5f}\n")
        del val_loader, val_loader_imagenet_r

        if evaluate_c:
            acc, rms = evaluate_variant(model,
                    args.batch_size_val, args.workers, args.imagenet_c_dir, args.save, variant="c")
            all_results["c_acc"] = acc
            all_results["c_rms"] = rms
        if evaluate_c_bar:
            acc, rms = evaluate_variant(model,
                    args.batch_size_val, args.workers, args.imagenet_c_bar_dir, args.save, variant="c_bar")
            all_results["c_bar_acc"] = acc
            all_results["c_bar_rms"] = rms
        if evaluate_p:
            flip_list, top5_list = test_imagenet_p(
                net=model,
                imagenet_path=args.imagenet_p_dir,
                classes_chosen=classes_chosen,
                save_path=os.path.join(args.save, "perturbation_results.csv"),
                batch_size=8,
                workers=args.workers,
            )
            print(f"Mean Flipping Prob\t{np.mean(flip_list):.5f}")
            print(f"Mean Top-5 Distance\t{np.mean(top5_list):.5f}")
            all_results["mean_flipping_prob"] = np.mean(flip_list)
            all_results["mean_top5_distance"] = np.mean(top5_list)

        print(all_results)
        write_to_json(all_results, os.path.join(args.save, f"all_results.json"))
        print("FINISHED EVALUATION")
        return

    ######################
    # Training
    ######################

    print(f"Using Dataset {args.data_standard}")
    train_data = ImageNetSubsetDataset(
        args.data_standard,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ])
    )
    mixing_set_transforms = [
        transforms.Resize(256),
        transforms.RandomCrop(224),
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

    mixing_set = datasets.ImageFolder(
        args.mixing_set,
        transform=transforms.Compose(mixing_set_transforms)
    )

    print("train_size: ", len(train_data))
    print("mixing_set_size: ", len(mixing_set))

    blending_fns = {
        "baseline"          : [],
        "augmix"            : [],
        "random_erase"      : [],
        "pixmix"            : [add, multiply],
        "ipmix"             : [add, multiply, random_pixels, random_elems],
        "layermix"          : [add, multiply, add, multiply, random_pixels, random_elems],
    }.get(args.mixing_type)

    assert blending_fns is not None, f"blending functions not found for {args.mixing_type}"
    print("using blending functions:", blending_fns)

    train_dataset = MixerDataset(
        dataset         = train_data,
        mixing_set      = mixing_set,
        depth           = args.depth,
        width           = args.width,
        image_aug_fns   = augs_image,
        spatial_aug_fns = augs_spatial,
        blending_fns    = blending_fns,
        magnitude       = args.magnitude,
        blending_ratio  = args.blending_ratio,
        mixer_type      = args.mixing_type,
        jsd             = args.jsd,
        normalize       = normalize
    )

    # Fix dataloader worker issue https://github.com/pytorch/pytorch/issues/5059
    def wif(id):
        uint64_seed = torch.initial_seed()
        ss = np.random.SeedSequence([uint64_seed])
        # More than 128 bits (4 32-bit words) would be overkill.
        np.random.seed(ss.generate_state(4))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, worker_init_fn=wif, drop_last=True)

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / (args.lr * args.batch_size / 256.)
        ))  # batch_size here is for 128 images TODO

    if args.start_epoch != 0:
        scheduler.step(args.start_epoch * len(train_loader))

    # ------- Main Training Loop

    if not args.resume:
        with open(os.path.join(args.save, "training_log.csv"), "w") as f:
            f.write("epoch,train_loss,train_acc1,train_acc5,val_loss,val_acc1,val_acc5,R_loss,R_acc1,R_acc5\n")

    for epoch in range(args.start_epoch, args.epochs):
        print(f"training epoch {epoch} at {datetime.now()}")
        # train for one epoch
        start_epoch = time.time()
        train_losses_avg, train_top1_avg, train_top5_avg = train(
            model, train_loader, criterion, optimizer, scheduler, epoch, args)
        print(f"Time spent in training: {(time.time() - start_epoch)/60:.4f}")
        print("Evaluating on validation set")
        val_losses_avg, val_top1_avg, val_top5_avg = validate(model, val_loader, criterion, args)

        # Save results in log file
        with open(os.path.join(args.save, "training_log.csv"), "a") as f:
            f.write("%03d,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f\n" % (
                (epoch + 1),
                train_losses_avg, train_top1_avg, train_top5_avg,
                val_losses_avg, val_top1_avg, val_top5_avg,
                0,0,0
                # val_R_losses_avg, val_R_top1_avg, val_R_top5_avg
            ))

        # remember best acc@1 and save checkpoint
        is_best = val_top1_avg > best_acc1
        best_acc1 = max(val_top1_avg, best_acc1)

        save_checkpoint({
            "epoch": epoch + 1,
            "model": args.model,
            "state_dict": model._orig_mod.state_dict(),
            "best_acc1": best_acc1,
            "optimizer": optimizer.state_dict(),
        }, is_best)
        print(f"time spent in training {epoch}: {(time.time() - start_epoch)/60:.4f}")
    # evaluate on imagenet_c and imagenet_c_bar
    evaluate_variant(model, args.batch_size, args.workers, args.imagenet_c_dir, args.save, variant="c")
    evaluate_variant(model, args.batch_size, args.workers, args.imagenet_c_bar_dir, args.save, variant="c_bar")


def evaluate_variant(model, batch_size, workers, c_dir, save_path, variant):
    assert variant in {"c", "c_bar"}
    model.eval()

    with open(os.path.join(save_path, f"eval_imagenet_{variant}_results.csv"), "a") as f:
        f.write("corruption,strength,top1_accuracy,calib\n")

    corruptions = [e for e in os.listdir(c_dir) if
                   os.path.isdir(os.path.join(c_dir, e))]  # All subdirectories, ignoring normal files
    corruptions = list(reversed(sorted(corruptions)))
    accuracy = []
    calibs = []
    for corr in corruptions:
        for strength in {1, 2, 3, 4, 5}:  # choose strengths
            dataloader = torch.utils.data.DataLoader(
                ImageNetSubsetDataset(
                    os.path.join(c_dir, corr, str(strength)),
                    transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize])),
                batch_size  = batch_size,
                shuffle     = False,
                num_workers = workers,
                pin_memory  = True
            )
            # Eval on this dataset
            acc, test_confidence, test_correct = get_net_results(dataloader, model)
            print(f"Eval on {corr} with strength {strength}: {acc}")

            curr_calib = calib_err(test_confidence, test_correct, p="2")

            with open(os.path.join(save_path, f"eval_imagenet_{variant}_results.csv"), "a") as f:
                f.write(f"{corr},{strength},{acc:.5f},{curr_calib:.5f}\n")

            accuracy.append(acc)
            calibs.append(curr_calib)
            del dataloader
    print(f"Accuracy on Imagenet-C: {100 * np.mean(accuracy):.3f}")
    print(f"RMS {100 * np.mean(calibs)}")

    return 100 * np.mean(accuracy), 100 * np.mean(calibs)


scaler = GradScaler()


def train(model, train_loader, criterion, optimizer, scheduler, epoch, args):
    # switch to train mode
    model.train()
    data_ema = 0.
    batch_ema = 0.
    loss_ema = 0.
    acc1_ema = 0.
    acc5_ema = 0.
    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time = time.time() - end
        optimizer.zero_grad(set_to_none=True)
        if not args.jsd:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
        else:
            images_all = torch.cat(images, 0).cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            logits_all = model(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(
                logits_all, images[0].size(0))

            # Cross-entropy is only computed on clean images
            loss = criterion(logits_clean, targets)

            p_clean, p_aug1, p_aug2 = F.softmax(
                logits_clean, dim=1), F.softmax(
                logits_aug1, dim=1), F.softmax(
                logits_aug2, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction="batchmean") +
                          F.kl_div(p_mixture, p_aug1, reduction="batchmean") +
                          F.kl_div(p_mixture, p_aug2, reduction="batchmean")) / 3.
            acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking

        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()
        # measure accuracy and record loss
        data_ema = data_ema * 0.1 + float(data_time) * 0.9
        batch_ema = batch_ema * 0.1 + float(batch_time) * 0.9
        loss_ema = loss_ema * 0.1 + float(loss) * 0.9
        acc1_ema = acc1_ema * 0.1 + float(acc1) * 0.9
        acc5_ema = acc5_ema * 0.1 + float(acc5) * 0.9

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if i % args.print_freq == 0:
            print(f"Batch {i:4d}/{len(train_loader)}: Data Time {data_ema:.3f} | Batch Time {batch_ema:.3f} | "
                  f"Train Loss {loss_ema:.3f} | LR {optimizer.param_groups[0]['lr']:.8f}",
                  f"Train Acc1 {acc1_ema:.3f} | Train Acc5 {acc5_ema:.3f}")
    return loss_ema, acc1_ema, batch_ema


@torch.inference_mode()
def validate(model, val_loader, criterion, args, r=False, a=False, adv=None):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    if adv:
        print("EVALUATING AGAINST ADVERSARY")
    elif r:
        print("EVALUATING ON IMAGENET-R")
    elif a:
        if args.num_classes == "200":  # the 200 classes are different
            return 0, 0, 0
        print("EVALUATING ON IMAGENET-A")

    confidence = []
    correct = []

    num_correct = 0

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # adversarial
        if adv:
            images = adv(model, images, target)

        # compute output
        output = model(images)
        if r and args.num_classes == "1000":  # eval on ImageNet-R
            output = output[:, imagenet_r_mask]
        elif a:
            output = output[:, imagenet_a_mask]

        loss = criterion(output, target)

        # accuracy
        pred = output.data.max(1)[1]
        num_correct += pred.eq(target.data).sum().item()

        confidence.extend(F.softmax(output, dim=1).max(1)[0].to("cpu").numpy().squeeze().tolist())
        pred = output.data.max(1)[1]
        correct.extend(pred.eq(target).to("cpu").numpy().squeeze().tolist())

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    # TODO: this should also be done with the ProgressMeter
    print(f"* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}")
    rms = calib_err(np.array(confidence.copy()), np.array(correct.copy()), p="2")
    print(f"RMS {100 * rms:.3f}")
    return losses.avg, top1.avg, top5.avg, rms


def save_checkpoint(state, is_best, filename=os.path.join(args.save, "model.pth.tar")):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save, "model_best.pth.tar"))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


@torch.inference_mode()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.inference_mode()
def get_net_results(dataloader, net):
    confidence = []
    correct = []

    num_correct = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()

        output = net(data)

        pred = output.data.max(1)[1]
        num_correct += pred.eq(target.data).sum().item()
        confidence.extend(F.softmax(output, dim=1).max(1)[0].to("cpu").numpy().squeeze().tolist())
        pred = output.data.max(1)[1]
        correct.extend(pred.eq(target).to("cpu").numpy().squeeze().tolist())

    return num_correct / len(dataloader.dataset), np.array(confidence.copy()), np.array(correct.copy())


if __name__ == "__main__":
    main()
