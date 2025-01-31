"""Condition img & prompt去生成圖片 (Eval)"""
"""python scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img <path-to-img.jpg> --strength 0.8"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# 切割File中的prompt (batch size)，之後用自己的dataset時，應該會用到。
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

# model = load_model_from_config(config, f"{opt.ckpt}"), load 權重及創建model, 注意! model.eval()
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}") # training iters?
    sd = pl_sd["state_dict"]
    
    # from ldm.util import instantiate_from_config
    model = instantiate_from_config(config.model)
    # teturn missing_keys, unexpected_keys
    m, u = model.load_state_dict(sd, strict=False)

    # verbose決定要不要囉唆
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

# 注意:這裡只能load一張
def load_img(path):
    # RGBA->RGB, Alpha通道->不透明度
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    
    # map() 會根據提供的函數對指定序列做映射。 
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    
    # 壓縮演算法替換為Image.Resampling.LANCZOS
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    
    image = np.array(image).astype(np.float32) / 255.0

    # 在前面多加一個維度，當作batch=1，且將顏色通道向前移。
    image = image[None].transpose(0, 3, 1, 2)

    # 轉成tensor
    image = torch.from_numpy(image)

    # range: 0~1 -> -1~1
    return 2.*image - 1.


def main():
    parser = argparse.ArgumentParser()

    # nargs="?" -> 引數只能是 0 個或是 1 個
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    # action='store_true' -> 將引數儲存為 boolean，若命令時有調用則true
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    # 將引數儲存為 boolean，若命令時有調用則true，否則false。 
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    # 使用 parse_args() 來從 parser 取得引數傳來的 Data
    opt = parser.parse_args()

    # 全部所有相同隨機種子設置，seed=42, from pytorch_lightning import seed_everything
    seed_everything(opt.seed)

    # 載入v1-inference.yaml，以及load pre-trained model. 用GPU跑
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        # 但是已經看到寫好了?
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    # 沒有"outputs/img2img-samples"話，創建(包括中間過程的)
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples

    # default=0, "rows in the grid (default: n_samples)"
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    # 指令載入prompt
    if not opt.from_file:
        prompt = opt.prompt
        ## if None, assertion error
        assert prompt is not None
        ## [['...', '...']]
        data = [batch_size * [prompt]]
    
    # 從file中載入prompt (自己寫file)
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            ## ['1', '2', '3', '4', '5', '6']
            data = f.read().splitlines()
            ## 形式: [('1', '2', '3'), ('4', '5', '6')]
            data = list(chunk(data, batch_size))

    # ouput圖片的儲存位置, "outputs/img2img-samples"
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    # base:  , grid:
    # 算目錄下多少檔案+子目錄
    # outpath = "outputs/img2img-samples", sample_path="outputs/img2img-samples/samples"
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # 是否為檔案非路徑
    assert os.path.isfile(opt.init_img)

    # init_image torch.Size([1,3,512,512])
    init_image = load_img(opt.init_img).to(device)

    # 將原先的一張img，repeat成數張(同一張)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

    # model = stable-diffusion/ldm/models/diffusion/ddpm.py, encode by vae
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'

    # t_enc:strength*50
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps") # target t_enc is 0 steps (if strength=0.01, then t_enc=0.5), (int(0.5)=0)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        # 如果scale=1，沒有unconditional_conditioning (考慮c:prompts, z_enc)(但為何需要uc?)
                        # scale越大，受prompt影響越大。
                        if opt.scale != 1.0:
                            # uc = unconditional_conditioning
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)

                        # -1~1 -> 0~1
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1
                        all_samples.append(x_samples)

                # 多張影像，用格線合成一張
                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)

                    # (n b) -> n*b
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
