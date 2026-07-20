import argparse
import os
import random

import numpy as np
import torch
import tqdm

import transferattack
from transferattack.utils import (
    AdvDataset,
    cnn_model_paper,
    load_pretrained_model,
    save_images,
    vit_model_paper,
    wrap_model,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate transferable adversarial examples with SimAttack."
    )
    parser.add_argument("-e", "--eval", action="store_true", help="evaluate generated adversarial images")
    parser.add_argument(
        "--attack",
        default="simattack",
        type=str,
        choices=transferattack.attack_zoo.keys(),
        help="attack algorithm name",
    )
    parser.add_argument("--epoch", default=10, type=int, help="number of optimization iterations")
    parser.add_argument("--transform_num", default=2000, type=int, help="number of random input transforms per iteration")
    parser.add_argument("--batchsize", default=128, type=int, help="batch size")
    parser.add_argument("--eps", default=16, type=float, help="epsilon budget in pixel scale, e.g. 16 means 16/255")
    parser.add_argument(
        "--alpha",
        default=1.6,
        type=float,
        help="kept for compatibility; this attack uses eps / epoch internally",
    )
    parser.add_argument("--model", default="resnet18", type=str, help="source surrogate model")
    parser.add_argument("--ensemble", action="store_true", help="enable comma-separated ensemble attack")
    parser.add_argument("--random_start", action="store_true", help="initialize perturbation randomly")
    parser.add_argument("--input_dir", default="./data", type=str, help="input data directory")
    parser.add_argument("--output_dir", default="./results", type=str, help="output directory")
    parser.add_argument("--targeted", action="store_true", help="run targeted attack")
    parser.add_argument("--GPU_ID", default="0", type=str, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--num_workers", default=4, type=int, help="DataLoader worker count")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID

    if not torch.cuda.is_available():
        raise RuntimeError("This implementation expects a CUDA GPU because the model and images are moved with .cuda().")

    seed_everything(args.seed)
    eps = args.eps / 255.0
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = AdvDataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        targeted=args.targeted,
        eval=args.eval,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if not args.eval:
        model_name = args.model.split(",") if args.ensemble or "," in args.model else args.model
        attacker = transferattack.load_attack_class(args.attack)(
            model_name=model_name,
            targeted=args.targeted,
            random_start=args.random_start,
            num_iter=args.epoch,
            transform_num=args.transform_num,
            epsilon=eps,
        )
        for images, labels, filenames in tqdm.tqdm(dataloader):
            images = images.cuda(non_blocking=True)
            perturbations = attacker(images, labels)
            save_images(args.output_dir, images.cpu() + perturbations.cpu(), filenames)
        return

    asr = {}
    res = "|"
    for model_name, model in load_pretrained_model(cnn_model_paper, vit_model_paper):
        model = wrap_model(model.eval().cuda())
        for param in model.parameters():
            param.requires_grad = False

        correct, total = 0, 0
        for images, labels, _ in dataloader:
            if args.targeted:
                labels = labels[1]
            images = images.cuda(non_blocking=True)
            pred = model(images)
            correct += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
            total += labels.shape[0]

        asr[model_name] = (correct / total) * 100 if args.targeted else (1 - correct / total) * 100
        print(model_name, asr[model_name])
        res += " {:.1f} |".format(asr[model_name])

    print(asr)
    print(res)
    with open("eval_results.txt", "a", encoding="utf-8") as f:
        f.write(args.output_dir + res + "\n")


if __name__ == "__main__":
    main()
