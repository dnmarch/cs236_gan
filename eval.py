import os
import pprint
import argparse

from tqdm import tqdm
import torch
from torchmetrics.image.fid import NoTrainInceptionV3

# styleGAN import
import util
from model import *

from models.GAN import Generator, Discriminator
from trainer import evaluate, prepare_data_for_gan, prepare_data_for_inception
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from models.GAN import Generator


def parse_args():
    r"""
    Parses command line arguments.
    """

    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(root_dir, "data"),
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to checkpoint used for evaluation.",
    )
    parser.add_argument(
        "--im_size",
        type=int,
        required=True,
        help=(
            "Images are resized to this resolution. "
            "Models are automatically selected based on resolution."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size used during evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device to evaluate on.",
    )
    parser.add_argument(
        "--submit",
        default=False,
        action="store_true",
        help="Generate Inception embeddings used for leaderboard submission.",
    )

    parser.add_argument("--start_depth", action="store", type=int, default=0,
                        help="Starting depth for training the network")

    parser.add_argument("--generator_file", action="store", type=str, default=None,
                        help="pretrained Generator file (compatible with my code)")
    parser.add_argument("--gen_shadow_file", action="store", type=str, default=None,
                        help="pretrained gen_shadow file")
    parser.add_argument("--discriminator_file", action="store", type=str, default=None,
                        help="pretrained Discriminator file (compatible with my code)")
    parser.add_argument("--gen_optim_file", action="store", type=str, default=None,
                        help="saved state of generator optimizer")
    parser.add_argument("--dis_optim_file", action="store", type=str, default=None,
                        help="saved_state of discriminator optimizer")

    return parser.parse_args()


def generate_submission(net_g, dataloader, nz, device, path="submission.pth"):
    r"""
    Generates Inception embeddings for leaderboard submission.
    """

    net_g.to(device).eval()
    inception = NoTrainInceptionV3(
        name="inception-v3-compat", features_list=["2048"]
    ).to(device)

    with torch.no_grad():
        real_embs, fake_embs = [], []
        for data, _ in tqdm(dataloader, desc="Generating Submission"):
            reals, z = prepare_data_for_gan(data, nz, device)
            fakes = net_g(z)
            reals = inception(prepare_data_for_inception(reals, device))
            fakes = inception(prepare_data_for_inception(fakes, device))
            real_embs.append(reals)
            fake_embs.append(fakes)
        real_embs = torch.cat(real_embs)
        fake_embs = torch.cat(fake_embs)
        embs = torch.stack((real_embs, fake_embs)).permute(1, 0, 2).cpu()

    torch.save(embs, path)


def eval(args):
    r"""
    Evaluates specified checkpoint.
    """

    # Set parameters
    nz, eval_size, num_workers = (
        128,
        4000 if args.submit else 10000,
        4,
    )

    # Configure models
    if args.im_size == 32:
        net_g = Generator32()
        net_d = Discriminator32()
    elif args.im_size == 64:
        net_g = Generator64()
        net_d = Discriminator64()
    else:
        raise NotImplementedError(f"Unsupported image size '{args.im_size}'.")


    # Loads checkpoint







    state_dict = torch.load(args.ckpt_path)
    net_g.load_state_dict(state_dict["net_g"])
    net_d.load_state_dict(state_dict["net_d"])

    # Configures eval dataloader
    _, eval_dataloader = util.get_dataloaders(
        args.data_dir, args.im_size, args.batch_size, eval_size, num_workers
    )

    if args.submit:
        # Generate leaderboard submission
        generate_submission(net_g, eval_dataloader, nz, args.device)

    else:
        # Evaluate models
        metrics = evaluate(net_g, net_d, eval_dataloader, nz, args.device)
        pprint.pprint(metrics)

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    print("Creating generator object ...")
    # create the generator object

    if args.generator_file is not None:
        gen = Generator(resolution=opt.dataset.resolution, num_channels=opt.dataset.channels, structure=opt.structure,
                        **opt.model.gen)
        dis = Discriminator(resolution=opt.dataset.resolution, num_channels=opt.dataset.channels,
                            structure=opt.structure, **opt.model.dis)
        gen.load_state_dict(torch.load(args.generator_file))
        dis.load_state_dict(torch.load(args.discriminator_file))
        metrics = evaluate(gen, dis, eval_dataloader, nz, args.device)
        pprint.pprint(metrics)

if __name__ == "__main__":
    eval(parse_args())
