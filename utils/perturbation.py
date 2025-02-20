# https://github.com/google-research/augmix/issues/5#issuecomment-583878711
import os

import numpy as np
import torch
from torchvision import transforms
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from imagenet_utils import VideoFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def dist(sigma, num_classes, mode='top5'):
    if mode == 'top5':
        cum_sum_top5 = np.cumsum(np.asarray([0] + [1] * 5 + [0] * (num_classes - 1 - 5)))
        return np.sum(np.abs(cum_sum_top5[:5] - cum_sum_top5[sigma-1][:5]))
    elif mode == 'zipf':
        recip = 1. / np.asarray(range(1, num_classes+1))
        return np.sum(np.abs(recip - recip[sigma-1]) * recip)


def ranking_dist(ranks, difficulty, noise_perturbation=False, mode='top5'):
    result = 0
    step_size = 1 if noise_perturbation else difficulty

    for vid_ranks in ranks:
        result_for_vid = []
        perm1 = vid_ranks[0]
        perm1_inv = np.argsort(perm1)

        for i in range(1, len(vid_ranks), step_size):
            perm2 = vid_ranks[i]
            result_for_vid.append(dist(perm2[perm1_inv], num_classes=len(ranks), mode=mode))

            if not noise_perturbation:
                perm1 = perm2
                perm1_inv = np.argsort(perm1)

        result += np.mean(result_for_vid) / len(ranks)
    return result


def flip_prob(predictions, difficulty, noise_perturbation=False):
    result = 0
    step_size = 1 if noise_perturbation else difficulty

    for vid_preds in predictions:
        result_for_vid = []
        prev_pred = vid_preds[0]

        for i in range(1, len(vid_preds), step_size):
            current_pred = vid_preds[i]
            result_for_vid.append(int(prev_pred != current_pred))

            if not noise_perturbation:
                prev_pred = current_pred
        result += np.mean(result_for_vid) / len(predictions)
    return result


@torch.inference_mode()
def calculate_p(net, path, perturbation_type, num_classes=10):
    np_data = np.float32(np.load(os.path.join(path, perturbation_type + '.npy')).transpose((0, 1, 4, 2, 3)))/255.0
    dataset = torch.from_numpy(np_data)
    p_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    # p_loader = torch.Size([10000, 31, 3, 32, 32])
    predictions, ranks = [], []
    for data in p_loader:
        num_vids, len_vids = data.shape[:2]  # bs -> 32
        data = data.view(-1, 3, 32, 32).cuda()  # num_vids * len_vids, 3, 32, 32
        output = net(data * 2 - 1)  # output ->  496, 100
        for vid in output.view(num_vids, len_vids, num_classes):
            predictions.append(vid.argmax(1).to('cpu').numpy())
            ranks.append(
                [np.uint16(rankdata(-frame, method='ordinal')) for frame in vid.to('cpu').numpy()])
    return predictions, ranks


def rankdata_ordinal(x):  # torch implementation of rank data function of scikit
    # @mughees this is experimental will add later.
    sorted_indices = torch.argsort(x, dim=1)
    ranks = torch.zeros_like(x, dtype=torch.int32, device=x.device)
    for i in range(x.size(0)):
        ranks[i, sorted_indices[i]] = torch.arange(1, x.size(1) + 1, dtype=torch.int32, device=x.device)
    return ranks


def test_p(net, path, num_classes=10, save_path = ""):
    flips = list()
    top5s = list()

    with open(save_path, 'a+') as file:
        file.write("Perturbation, Flipping Prob, Top5 Distance \n")
    for p in [
        "gaussian_noise",
        "shot_noise",
        "motion_blur",
        "zoom_blur",
        # "snow",
        "spatter",
        "brightness",
        "translate",
        "rotate",
        "tilt",
        "scale"
    ]:  # ,"speckle_noise", "gaussian_blur", "snow" "spatter", "shear"]:

        # @mughees: in the code it shows they used spatter noise instead of snow but in the released images,
        # snow is in main folder and spatter is in extra. but lets use spatter for now.
        # I will evaluate their released models too.
        # https://drive.google.com/drive/folders/1Tpssw4Vn6X_4hmIW8KK5LzkKCZpAaae8

        predictions, ranks = calculate_p(net, path, p, num_classes)
        ranks = np.asarray(ranks)
        difficulty=1
        current_flip = flip_prob(
            predictions,
            difficulty=difficulty,
            noise_perturbation="noise" in p)
        flips.append(current_flip)
        # current_zipf = ranking_dist(
        #     ranks,
        #     difficulty=difficulty,
        #     noise_perturbation=True if 'noise' in p else False,
        #     mode='zipf')
        # zipf_list.append(current_zipf)
        current_top5 = ranking_dist(
                ranks,
                difficulty=difficulty,
                noise_perturbation=True if 'noise' in p else False,
                mode='top5')
        top5s.append(current_top5)
        print(f"{p}\tFlipping Prob\t{current_flip * 100 :.3f}")
        print(f"{p}\tTop5 Distance\t{current_top5:.5f}")
        with open(save_path, 'a+') as file:
            file.write(f"{p}, {current_flip * 100 :.3f}, {current_top5:.5f} \n")
    print(f"Mean Flipping Prob\t{np.mean(flips)*100:.3f}")
    print(f"Mean T5D \t{np.mean(top5s):.3f}")
    return flips, top5s


@torch.inference_mode()
def calculate_imagenet_p(net, path, perturbation_type, classes_chosen: list[str], batch_size=32, workers=8):
    loader = torch.utils.data.DataLoader(
        VideoFolder(root=str(os.path.join(path, perturbation_type)),
                    classes_chosen=classes_chosen,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    predictions, ranks = [], []
    for idx, (data, target) in enumerate(loader):
        num_vids = data.size(0)
        data = data.view(-1, 3, 224, 224).cuda()
        output = net(data)
        for vid in output.view(num_vids, -1, len(classes_chosen)):
            predictions.append(vid.argmax(1).to('cpu').numpy())
            ranks.append([np.uint16(rankdata(-frame, method='ordinal')) for frame in vid.to('cpu').numpy()])
        if idx % 100 == 0:
            print(f"Batch {idx} done")
    print(f"Done with {perturbation_type}!")

    return predictions, ranks


def test_imagenet_p(net, imagenet_path, classes_chosen, save_path = "", batch_size=32, workers=8):
    flips = list()
    top5s = list()
    with open(save_path, 'a+') as file:
        file.write("Perturbation, Flipping Prob, Top5 Distance \n")

    for p in [
        "gaussian_noise",
        "shot_noise",
        "motion_blur",
        "zoom_blur",
        "spatter",
        "brightness",
        "translate",
        "rotate",
        "tilt",
        "scale",
        "speckle_noise",
        "gaussian_blur",
        "snow",
        "shear"
    ]:
        predictions, ranks = calculate_imagenet_p(net, imagenet_path, p, classes_chosen, batch_size=batch_size, workers=workers)
        ranks = np.asarray(ranks)
        current_flip = flip_prob(
            predictions,
            difficulty=1,
            noise_perturbation="noise" in p)
        flips.append(current_flip)
        current_top5 = ranking_dist(
            ranks,
            difficulty=1,
            noise_perturbation="noise" in p,
            mode='top5')
        top5s.append(current_top5)
        print(f"{p}\tFlipping Prob\t{current_flip * 100 :.3f}")
        print(f"{p}\tTop5 Distance\t{current_top5:.5f}")
        with open(save_path, 'a+') as file:
            file.write(f"{p}, {current_flip * 100 :.3f}, {current_top5:.5f} \n")
    print(f"Mean Flipping Prob\t{np.mean(flips)*100:.3f}")
    print(f"Mean T5D \t{np.mean(top5s):.3f}")
    return flips, top5s