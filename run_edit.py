import gc
import io
import math
import sys

from PIL import Image, ImageOps
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

import numpy as np

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from dalle_pytorch import DiscreteVAE, VQGanVAE

from einops import rearrange
from math import log2, sqrt

import argparse
import pickle

import os

from encoders.modules import BERTEmbedder


################################### mask_fusion ######################################
from util.metrics_accumulator import MetricsAccumulator
metrics_accumulator = MetricsAccumulator()

from pathlib import Path
from PIL import Image
################################### mask_fusion ######################################



import clip
import lpips
from torch.nn.functional import mse_loss

################################### CLIPseg ######################################
from models.clipseg import CLIPDensePredT
from  torchvision import utils as vutils
import cv2  

segmodel = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
segmodel.eval()

# non-strict, because we only stored decoder weights (not CLIP weights)
segmodel.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)



################################### CLIPseg ######################################

def str2bool(x):
    return x.lower() in ('true')
    
# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default = 'finetune.pt',
                   help='path to the diffusion model')

parser.add_argument('--kl_path', type=str, default = 'kl-f8.pt',
                   help='path to the LDM first stage model')

parser.add_argument('--bert_path', type=str, default = 'bert.pt',
                   help='path to the LDM first stage model')

parser.add_argument('--text', type = str, required = False, default = '',
                    help='your text prompt')

parser.add_argument('--edit', type = str, required = False,
                    help='path to the image you want to edit (either an image file or .npy containing a numpy array of the image embeddings)')

parser.add_argument('--edit_x', type = int, required = False, default = 0,
                    help='x position of the edit image in the generation frame (need to be multiple of 8)')

parser.add_argument('--edit_y', type = int, required = False, default = 0,
                    help='y position of the edit image in the generation frame (need to be multiple of 8)')

parser.add_argument('--edit_width', type = int, required = False, default = 0,
                    help='width of the edit image in the generation frame (need to be multiple of 8)')

parser.add_argument('--edit_height', type = int, required = False, default = 0,
                    help='height of the edit image in the generation frame (need to be multiple of 8)')

parser.add_argument('--mask', type = str, required = False,
                    help='path to a mask image. white pixels = keep, black pixels = discard. width = image width/8, height = image height/8')

parser.add_argument('--negative', type = str, required = False, default = '',
                    help='negative text prompt')

parser.add_argument('--skip_timesteps', type=int, required = False, default = 0,
                   help='how many diffusion steps are gonna be skipped')

parser.add_argument('--prefix', type = str, required = False, default = '',
                    help='prefix for output files')

parser.add_argument('--num_batches', type = int, default = 1, required = False,
                    help='number of batches')

parser.add_argument('--batch_size', type = int, default = 1, required = False,
                    help='batch size')

parser.add_argument('--width', type = int, default = 256, required = False,
                    help='image size of output (multiple of 8)')

parser.add_argument('--height', type = int, default = 256, required = False,
                    help='image size of output (multiple of 8)')

parser.add_argument('--seed', type = int, default=-1, required = False,
                    help='random seed')

parser.add_argument('--guidance_scale', type = float, default = 5.0, required = False,
                    help='classifier-free guidance scale')

parser.add_argument('--steps', type = int, default = 100, required = False,
                    help='number of diffusion steps')

parser.add_argument('--cpu', dest='cpu', action='store_true')

parser.add_argument('--clip_score', dest='clip_score', action='store_true')

parser.add_argument('--clip_guidance', dest='clip_guidance', action='store_true')

parser.add_argument('--clip_guidance_scale', type = float, default = 150, required = False,
                    help='Controls how much the image should look like the prompt') # may need to use lower value for ddim

parser.add_argument('--cutn', type = int, default = 16, required = False,
                    help='Number of cuts')

parser.add_argument('--ddim', dest='ddim', action='store_true') # turn on to use 50 step ddim

parser.add_argument('--ddpm', dest='ddpm', action='store_true') # turn on to use 50 step ddim

parser.add_argument("-bg","--background", type=str, help="The path to the background edit with", default=None)

parser.add_argument("-fp", "--fromtext", type=str, help="from text prompt", default=None)

parser.add_argument('--lpips_sim_lambda', type=float, default=1000) # The LPIPS similarity to the input image

parser.add_argument("--background_preservation_loss", help="Indicator for using the background preservation loss", action="store_true", default=True)

parser.add_argument("--l2_sim_lambda", type=float, help="The L2 similarity to the input image", default=10000)

parser.add_argument('--smooth_weight', type=int, default=1, help='Weight for boundary smoothness')

parser.add_argument('--use_smooth_loss', type=str2bool, default=True, help='use boundary smoothness loss')

parser.add_argument("--no_enforce_background", help="Indicator disabling the last background enforcement", action="store_false", dest="enforce_background",)


args = parser.parse_args()

Path("./output/"+args.prefix).mkdir(parents=True, exist_ok=True)

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()

        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)






device = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
print('Using device:', device)

model_state_dict = torch.load(args.model_path, map_location='cpu')

model_params = {
    'attention_resolutions': '32,16,8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': args.steps,  # Modify this value to decrease the number of
                                 # timesteps.
    'image_size': 32,
    'learn_sigma': False,
    'noise_schedule': 'linear',
    'num_channels': 320,
    'num_heads': 8,
    'num_res_blocks': 2,
    'resblock_updown': False,
    'use_fp16': False,
    'use_scale_shift_norm': False,
    'clip_embed_dim': 768 if 'clip_proj.weight' in model_state_dict else None,
    'image_condition': True if model_state_dict['input_blocks.0.0.weight'].shape[1] == 8 else False,
    'super_res_condition': True if 'external_block.0.0.weight' in model_state_dict else False,
}

if args.ddpm:
    model_params['timestep_respacing'] = '1000'
if args.ddim:
    if args.steps:
        model_params['timestep_respacing'] = 'ddim'+str(args.steps)
    else:
        model_params['timestep_respacing'] = 'ddim50'
elif args.steps:
    model_params['timestep_respacing'] = str(args.steps)

model_config = model_and_diffusion_defaults()
model_config.update(model_params)

if args.cpu:
    model_config['use_fp16'] = False


model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(model_state_dict, strict=False)
model.requires_grad_(args.clip_guidance).eval().to(device)

if model_config['use_fp16']:
    model.convert_to_fp16()
else:
    model.convert_to_fp32()

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


lpips_model = lpips.LPIPS(net="vgg").to(device)


ldm = torch.load(args.kl_path, map_location="cpu")
ldm.to(device)
ldm.eval()
ldm.requires_grad_(args.clip_guidance)
set_requires_grad(ldm, args.clip_guidance)

bert = BERTEmbedder(1280, 32)
sd = torch.load(args.bert_path, map_location="cpu")
bert.load_state_dict(sd)

bert.to(device)
bert.half().eval()
set_requires_grad(bert, False)


clip_model, clip_preprocess = clip.load('ViT-L/14', device=device, jit=False)
clip_model.eval().requires_grad_(False)
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])



input_image = Image.open(args.edit).convert("RGB")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((256, 256)),
])
img = transform(input_image).unsqueeze(0)

with torch.no_grad():
    preds = segmodel(img.repeat(1,1,1,1), args.fromtext)[0]


mask = torch.sigmoid(preds[0][0])
vutils.save_image(mask, args.mask, normalize=True)
image = cv2.imread(args.mask)                     
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
	


ret,thresh = cv2.threshold(image, 100, 255, cv2.THRESH_TRUNC, image) 
timg = np.array(thresh)
print(timg)
x, y = timg.shape
for row in range(x):
    for col in range(y):
        if (timg[row][col]) == 100:
            timg[row][col] = 255
        if (timg[row][col]) < 100:
            timg[row][col] = 0
cv2.imwrite(args.mask, timg) 



if args.background is None:
    fulltensor = torch.full_like(mask,fill_value=255)

    bgtensor = fulltensor-timg
    vutils.save_image(bgtensor, args.mask, normalize=True)
    



def unaugmented_clip_distance(self, x, text_embed):
    x = F.resize(x, [self.clip_size, self.clip_size])
    image_embeds = self.clip_model.encode_image(x).float()
    dists = spherical_dist_loss(image_embeds, text_embed)

    return dists.item()



def do_run():
    if args.seed >= 0:
        torch.manual_seed(args.seed)


    text_emb = bert.encode([args.text]*args.batch_size).to(device).float()
    text_blank = bert.encode([args.negative]*args.batch_size).to(device).float()

    text = clip.tokenize([args.text]*args.batch_size, truncate=True).to(device)
    text_clip_blank = clip.tokenize([args.negative]*args.batch_size, truncate=True).to(device)



    text_emb_clip = clip_model.encode_text(text)
    text_emb_clip_blank = clip_model.encode_text(text_clip_blank)
    make_cutouts = MakeCutouts(clip_model.visual.input_resolution, args.cutn)
    text_emb_norm = text_emb_clip[0] / text_emb_clip[0].norm(dim=-1, keepdim=True)
    image_embed = None


    if args.edit:
        if args.edit.endswith('.npy'):
            with open(args.edit, 'rb') as f:
                im = np.load(f)
                im = torch.from_numpy(im).unsqueeze(0).to(device)

                input_image = torch.zeros(1, 4, args.height//8, args.width//8, device=device)

                y = args.edit_y//8
                x = args.edit_x//8

                ycrop = y + im.shape[2] - input_image.shape[2]
                xcrop = x + im.shape[3] - input_image.shape[3]

                ycrop = ycrop if ycrop > 0 else 0
                xcrop = xcrop if xcrop > 0 else 0

                input_image[0,:,y if y >=0 else 0:y+im.shape[2],x if x >=0 else 0:x+im.shape[3]] = im[:,:,0 if y > 0 else -y:im.shape[2]-ycrop,0 if x > 0 else -x:im.shape[3]-xcrop]

                input_image_pil = ldm.decode(input_image)
                input_image_pil = TF.to_pil_image(input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1))

                input_image *= 0.18215
        else:
            w = args.edit_width if args.edit_width else args.width
            h = args.edit_height if args.edit_height else args.height

            input_image_pil = Image.open(fetch(args.edit)).convert('RGB')

            init_image_pil = input_image_pil.resize((args.height, args.width), Image.Resampling.LANCZOS)

            input_image_pil = ImageOps.fit(input_image_pil, (w, h))

            input_image = torch.zeros(1, 4, args.height//8, args.width//8, device=device)

            im = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device)
            
            init_image = (TF.to_tensor(init_image_pil).to(device).unsqueeze(0).mul(2).sub(1))

            im = 2*im-1
            im = ldm.encode(im).sample()

            y = args.edit_y//8
            x = args.edit_x//8

            input_image = torch.zeros(1, 4, args.height//8, args.width//8, device=device)

            ycrop = y + im.shape[2] - input_image.shape[2]
            xcrop = x + im.shape[3] - input_image.shape[3]

            ycrop = ycrop if ycrop > 0 else 0
            xcrop = xcrop if xcrop > 0 else 0

            input_image[0,:,y if y >=0 else 0:y+im.shape[2],x if x >=0 else 0:x+im.shape[3]] = im[:,:,0 if y > 0 else -y:im.shape[2]-ycrop,0 if x > 0 else -x:im.shape[3]-xcrop]

            input_image_pil = ldm.decode(input_image)
            input_image_pil = TF.to_pil_image(input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1))

            input_image *= 0.18215

        if args.mask:
            mask_image = Image.open(fetch(args.mask)).convert('L')
            mask_image = mask_image.resize((args.width//8,args.height//8), Image.Resampling.LANCZOS)
            mask = transforms.ToTensor()(mask_image).unsqueeze(0).to(device)

        mask1 = (mask > 0.5)
        mask1 = mask1.float()

        input_image *= mask1

        image_embed = torch.cat(args.batch_size*2*[input_image], dim=0).float()
    elif model_params['image_condition']:
        # using inpaint model but no image is provided
        image_embed = torch.zeros(args.batch_size*2, 4, args.height//8, args.width//8, device=device)

    kwargs = {
        "context": torch.cat([text_emb, text_blank], dim=0).float(),
        "clip_embed": torch.cat([text_emb_clip, text_emb_clip_blank], dim=0).float() if model_params['clip_embed_dim'] else None,
        "image_embed": image_embed
    }

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + args.guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    cur_t = None

    def cond_fn(x, t, context=None, clip_embed=None, image_embed=None):
        with torch.enable_grad():
            x = x[:args.batch_size].detach().requires_grad_()

            n = x.shape[0]

            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t

            kw = {
                'context': context[:args.batch_size],
                'clip_embed': clip_embed[:args.batch_size] if model_params['clip_embed_dim'] else None,
                'image_embed': image_embed[:args.batch_size] if image_embed is not None else None
            }

            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs=kw)

            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)

            x_in /= 0.18215

            x_img = ldm.decode(x_in)

            clip_in = normalize(make_cutouts(x_img.add(1).div(2)))
            clip_embeds = clip_model.encode_image(clip_in).float()
            dists = spherical_dist_loss(clip_embeds.unsqueeze(1), text_emb_clip.unsqueeze(0))
            dists = dists.view([args.cutn, n, -1])

            losses = dists.sum(2).mean(0)

            loss = losses.sum() * args.clip_guidance_scale
        

            if args.background_preservation_loss:
                if mask is not None:
                    if mask is not None:
                        masked_background = x_in * (1 - mask)
                    else:
                        masked_background = x_in
                if args.lpips_sim_lambda:
                            loss = (
                                loss
                                + lpips_model(masked_background, args.edit).sum()
                                * args.lpips_sim_lambda
                            )
                if args.l2_sim_lambda:
                        loss = (
                            loss
                            + mse_loss(masked_background, args.edit) * args.l2_sim_lambda
                        )
            return -torch.autograd.grad(loss, x)[0]



    @torch.no_grad()
    def postprocess_fn(out, t):
        if mask is not None:
            background_stage_t = diffusion.q_sample(init_image, t[0])
            background_stage_t = torch.tile(
                background_stage_t, dims=(args.batch_size, 1, 1, 1)
            )
            out["sample"] = out["sample"] * mask + background_stage_t * (1 - mask)
        return out

    if args.ddpm:
        sample_fn = diffusion.p_sample_loop_progressive
    elif args.ddim:
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    def save_sample(i, sample, clip_score=True):
        for k, image in enumerate(sample['pred_xstart'][:args.batch_size]):
            image /= 0.18215
            im = image.unsqueeze(0)
            out = ldm.decode(im)
            metrics_accumulator.print_average_metric()

            for b in range(args.batch_size):
                pred_image = sample["pred_xstart"][b]

                if (
                    args.mask is not None
                    and args.enforce_background
                ):
                    mask_image = Image.open(fetch(args.mask)).convert('L')
                    mask_image = mask_image.resize((args.height, args.width), Image.Resampling.LANCZOS)
                    mask = transforms.ToTensor()(mask_image).unsqueeze(0).to(device)
                    pred_image = (
                        init_image[0] * mask[0] + out * (1 - mask[0])
                    )

                pred_image_pil = TF.to_pil_image(pred_image.squeeze(0).add(1).div(2).clamp(0, 1))
                ranked_pred_path= f'output/{args.prefix}{i * args.batch_size + k:02}.png'
                pred_image_pil.save(ranked_pred_path)
            


    for i in range(args.num_batches):
        cur_t = diffusion.num_timesteps - 1

        samples = sample_fn(
            model_fn,
            (args.batch_size*2, 4, int(args.height/8), int(args.width/8)),
            clip_denoised=False,
            model_kwargs=kwargs,
            cond_fn=cond_fn if args.clip_guidance else None,
            device=device,
            progress=True,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            if j % 5 == 0 and j != diffusion.num_timesteps - 1:
                save_sample(i, sample)
        save_sample(i, sample, args.clip_score)

gc.collect()
do_run()
