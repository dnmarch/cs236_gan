import os
import argparse
import numpy as np
from PIL import Image

import torch

from models.GAN import Generator
from generate_grid import adjust_dynamic_range



def draw_style_mixing(png, gen, out_depth, n_row=6):
    # assume mixing of three images
    with torch.no_grad():
        latent_size = gen.g_mapping.latent_size
        z0 = np.random.RandomState(1).randn(n_row, latent_size)
        z1 = np.random.RandomState(2).randn(n_row, latent_size)
        z2 = np.random.RandomState(3).randn(n_row, latent_size)
        zs = [z0, z1, z2]
        ws = []
        images = []
        w = h = 2 ** (out_depth + 2)
        for z in zs:
            z = torch.from_numpy(z.astype(np.float32))
            # w: 1, 10, 512
            ws.append(gen.g_mapping(z))
            images.append(gen.g_synthesis(ws[-1], depth=out_depth, alpha=1))
        # generate mix images
        w_mix = torch.cat([ws[0][:, :4, :], ws[1][:, 4:8, :], ws[1][:, 8:, :]], axis=1)
        print(w_mix.shape, ws[-1].shape)
        # image: n_row, 3, 64, 64
        images.append(gen.g_synthesis(w_mix, depth=out_depth, alpha=1))
        print(len(images))
        canvas = Image.new('RGB', (w * (4 + 1), h * (n_row + 1)), 'white')
        print(images[-1].shape)

        for row in range(n_row):
            for col in range(4):
                img = images[col][row]
                img = adjust_dynamic_range(img)
                img = img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                canvas.paste(Image.fromarray(img, 'RGB'), ((col + 1) * w, (row + 1) * h))
        print(png)
        canvas.save(png)


def draw_style_mixing_figure(png, gen, out_depth, src_seeds, dst_seeds, style_ranges):
    n_col = len(src_seeds)
    n_row = len(dst_seeds)
    w = h = 2 ** (out_depth + 2)
    with torch.no_grad():
        latent_size = gen.g_mapping.latent_size
        src_latents_np = np.stack([np.random.RandomState(seed).randn(latent_size, ) for seed in src_seeds])
        dst_latents_np = np.stack([np.random.RandomState(seed).randn(latent_size, ) for seed in dst_seeds])
        src_latents = torch.from_numpy(src_latents_np.astype(np.float32))
        dst_latents = torch.from_numpy(dst_latents_np.astype(np.float32))
        print(torch.min(src_latents), torch.max(src_latents))
        src_dlatents = gen.g_mapping(src_latents)  # [seed, layer, component]
        dst_dlatents = gen.g_mapping(dst_latents)  # [seed, layer, component]
        src_images = gen.g_synthesis(src_dlatents, depth=out_depth, alpha=1)
        dst_images = gen.g_synthesis(dst_dlatents, depth=out_depth, alpha=1)

        src_dlatents_np = src_dlatents.numpy()
        dst_dlatents_np = dst_dlatents.numpy()
        canvas = Image.new('RGB', (w * (n_col + 1), h * (n_row + 1)), 'white')
        for col, src_image in enumerate(list(src_images)):
            src_image = adjust_dynamic_range(src_image)
            src_image = src_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
        for row, dst_image in enumerate(list(dst_images)):
            dst_image = adjust_dynamic_range(dst_image)
            dst_image = dst_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))

            row_dlatents = np.stack([dst_dlatents_np[row]] * n_col)
            row_dlatents[:, style_ranges[row]] = src_dlatents_np[:, style_ranges[row]]
            row_dlatents = torch.from_numpy(row_dlatents)

            row_images = gen.g_synthesis(row_dlatents, depth=out_depth, alpha=1)
            for col, image in enumerate(list(row_images)):
                image = adjust_dynamic_range(image)
                image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                canvas.paste(Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
        canvas.save(png)


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    print("Creating generator object ...")
    # create the generator object
    gen = Generator(resolution=opt.dataset.resolution,
                    num_channels=opt.dataset.channels,
                    structure=opt.structure,
                    **opt.model.gen)

    print("Loading the generator weights from:", args.generator_file)
    # load the weights into it
    gen.load_state_dict(torch.load(args.generator_file))

    # path for saving the files:
    # generate the images:
    # src_seeds = [639, 701, 687, 615, 1999], dst_seeds = [888, 888, 888],

    draw_style_mixing(args.save_path, gen, out_depth=args.depth)
    draw_style_mixing_figure(os.path.join('figure03-style-mixing.png'), gen,
                             out_depth=args.depth, src_seeds=[639, 1995, 687, 615, 1999], dst_seeds=[888, 888, 888],
                             style_ranges=[range(0, 2)] * 1 + [range(2, 8)] * 1 + [range(8, 14)] * 1)


    print('Done.')


def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./configs/sample_race_256.yaml')
    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", required=True)
    parser.add_argument("--depth", action="store", type=int, default=4,
                        help="depth for training the network")
    parser.add_argument('--save_path', default='/content/cs236_gan/style_mix.png')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_arguments())
