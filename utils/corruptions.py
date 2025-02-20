import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.calibration import calib_err

CORRUPTIONS = [
    "gaussian_noise"   , "shot_noise" , "impulse_noise"   , "defocus_blur",
    "glass_blur"       , "motion_blur", "zoom_blur"       , "snow"        ,
    "frost"            , "fog"        , "brightness"      , "contrast"    ,
    "elastic_transform", "pixelate"   , "jpeg_compression"                ]

CBAR_CORRUPTIONS = [
    "blue_noise_sample", "brownish_noise", "checkerboard_cutout" , "inverse_sparkles",
    "pinch_and_twirl"  , "ripple"        , "circular_motion_blur", "lines"           ,
    "sparkles"         , "transverse_chromatic_abberation"]

@dataclass
class TestOut:
    loss: float
    acc: float
    confidence: np.ndarray
    correct: np.ndarray
    total_correct: float

@torch.no_grad()
def test(net, test_loader, adv=None) -> TestOut:
    net.eval()
    confidence = []
    correct = []
    total_loss = 0.
    total_correct = 0
    num_correct = 0

    for images, targets in test_loader:
        images, targets = images.cuda(), targets.cuda()
        # adversarial
        if adv:
            images = adv(net, images, targets)
        logits = net(images)
        pred = logits.data.max(1)[1]
        num_correct += pred.eq(targets.data).sum().item()

        loss = F.cross_entropy(logits, targets)
        conf = F.softmax(logits, dim=1).max(1)[0]

        total_loss += float(loss.data)
        total_correct += pred.eq(targets.data).sum().item()
        confidence.extend(conf.data.to('cpu').numpy().squeeze().tolist())  # i think np.stack can be used
        correct.extend(pred.eq(targets).to('cpu').numpy().squeeze().tolist())
    return TestOut(
        loss          = total_loss / len(test_loader),
        acc           = num_correct / len(test_loader.dataset),
        confidence    = np.array(confidence),
        correct       = np.array(correct),
        total_correct = total_correct / len(test_loader.dataset)
    )


@dataclass
class TestC:
    acc: float
    rms: float


def test_c(net, test_data, base_path, eval_batch_size=512, workers=4, save_path=None) -> TestC:
    corruption_accs = []
    corruption_rms_errors = []
    corrs = CBAR_CORRUPTIONS if 'Bar' in base_path else CORRUPTIONS
    file_name = "corruption_bar_results.txt" if 'Bar' in base_path else "corruption_results.txt"
    file = open(os.path.join(save_path, file_name), "w")
    file.write("Corruption, Test Loss, Test Error\n")

    for corruption in corrs:
        # Reference to original data is mutated
        # todo @mughees replace it with torch tensor
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
        test_loader = DataLoader(
            test_data,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True)
        ctx = test(net, test_loader)

        corruption_accs.append(ctx.acc)
        corruption_rms_errors.append(calib_err(ctx.confidence, ctx.correct, p='2'))
        print(f"{corruption}\tTest Loss {ctx.loss:.3f} | Test Error {100 - 100. * ctx.acc:.3f}"
              f" | RMS Error {100 * corruption_rms_errors[-1]:.3f}")
        file.write(f"{corruption}, {ctx.loss:.3f}, {100 - 100. * ctx.acc:.3f}\n")

    file.write(f"Mean Corruption Error: {100 - 100. * np.mean(corruption_accs)}")
    print(f"RMS: {np.mean(corruption_rms_errors)}")

    file.close()
    return TestC(
        acc=np.mean(corruption_accs).item(),
        rms=np.mean(corruption_rms_errors).item())
