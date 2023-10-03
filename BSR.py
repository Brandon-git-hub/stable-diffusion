import torch, torchvision
from einops import rearrange, repeat
import numpy as np
from PIL import Image
from notebook_helpers import get_model, run


def load_img_dict(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size

    example = dict()
    resolution = 256
    up_f = 4
    left = (w-resolution)/2
    top = (h-resolution)/2
    right = left + resolution
    bottom = top + resolution
    c = im.crop((left, top, right, bottom))
    w, h = c.size
    print('Input img cropped into ',w,h)

    c = np.array(c).astype(np.float32) / 255.0
    c = c[None].transpose(0, 3, 1, 2)
    c = torch.from_numpy(c)

    # c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
    c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[2], up_f * c.shape[3]], antialias=True)
    c_up = rearrange(c_up, '1 c h w -> 1 h w c')
    c = rearrange(c, '1 c h w -> 1 h w c')
    c = 2. * c - 1.

    c = c.to(torch.device("cuda:0"))
    example["LR_image"] = c
    example["image"] = c_up
    print(f'LR_image: {c.shape},{c.dtype}, Up_image: {c_up.shape},{c_up.dtype}')

    return example



def main():
    mode = 'superresolution'
    custom_steps = int(100)
    img_dict = load_img_dict('/media/user/VCLAB/DL/stable-diffusion/assets/deblur.png')
    model = get_model(mode)

    # 
    logs = run(model=model["model"], example=img_dict, custom_steps=custom_steps)

    sample = logs["sample"]
    sample = sample.detach().cpu()
    sample = torch.clamp(sample, -1., 1.)
    sample = (sample + 1.) / 2. * 255
    sample = sample.numpy().astype(np.uint8)
    sample = np.transpose(sample, (0, 2, 3, 1))
    print(sample.shape)
    output_img = Image.fromarray(sample[0])

    output_img.save('./outputs/BSR-samples/0.png')


if __name__=='__main__':
    main()
