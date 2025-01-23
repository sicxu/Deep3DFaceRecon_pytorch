"""This script is the test script for Deep3DFaceRecon_pytorch"""

import os
import os.path as osp

import numpy as np
import torch
from PIL import Image

from models import create_model
from options.test_options import TestOptions
from util.load_mats import load_lm3d
from util.preprocess import align_img
from util.visualizer import MyVisualizer


def get_data_path(root="examples"):
    supported_extensions = list(Image.registered_extensions().keys())
    # print(f"--> supported_extensions: {supported_extensions}")

    file_list = [
        ff
        for ff in sorted(os.listdir(root))
        if osp.splitext(ff)[-1].lower() in supported_extensions
    ]
    im_path = [osp.join(root, ff) for ff in file_list]
    lm_path = [
        osp.join(root, "detections", osp.splitext(ff)[0] + ".txt") for ff in file_list
    ]

    return im_path, lm_path


def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB
    im = Image.open(im_path).convert("RGB")
    W, H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = (
            torch.tensor(np.array(im) / 255.0, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm


def main(opt):
    if opt.device == "mps":
        assert torch.mps.is_available(), "MPS is not available"
        device = torch.device("mps")
        print("--> Running on mps")
    elif opt.device == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available"
        device = torch.device(opt.gpu_ids[0])
        torch.cuda.set_device(device)
        print(f"--> Running on cuda:{opt.gpu_ids[0]}")
    else:
        device = torch.device("cpu")
        print("--> Running on cpu")

    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()

    if opt.do_visualize:
        assert (
            model.renderer is not None
        ), "Visualization is only supported for models with a renderer"

        visualizer = MyVisualizer(opt)
        save_dir = osp.join(
            visualizer.img_dir,
            opt.img_folder.split(osp.sep)[-1],
            "epoch_%s_%06d" % (opt.epoch, 0),
        )
    else:
        save_dir = opt.img_folder + "-results"
    print(f"--> Save dir: {save_dir}")

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    im_path, lm_path = get_data_path(opt.img_folder)
    lm3d_std = load_lm3d(opt.bfm_folder)

    for i in range(len(im_path)):
        print(i, im_path[i])
        img_name = osp.splitext(osp.basename(im_path[i]))[0]

        if not osp.isfile(lm_path[i]):
            print("%s is not found !!!" % lm_path[i])
            continue
        im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
        data = {
            "imgs": im_tensor,
            "lms": lm_tensor,
        }
        model.set_input(data)  # unpack data from data loader
        model.test(do_render=opt.do_visualize)  # run inference

        if opt.do_visualize:
            visuals = model.get_current_visuals()  # get image results
            visualizer.display_current_results(
                visuals,
                0,
                opt.epoch,
                dataset=osp.basename(opt.img_folder),
                save_results=True,
                count=i,
                name=img_name,
                add_image=False,
            )

        model.save_mesh(
            osp.join(save_dir, img_name + ".obj")
        )  # save reconstruction meshes

        model.save_coeff(
            osp.join(save_dir, img_name + ".mat")
        )  # save predicted coefficients

    print(f"--> Results saved under dir: {save_dir}")


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    main(opt)
