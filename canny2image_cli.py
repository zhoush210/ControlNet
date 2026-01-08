from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random
import argparse
from PIL import Image

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def main():
    parser = argparse.ArgumentParser(description='ControlNet Canny to Image')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input image path')
    parser.add_argument('--prompt', '-p', type=str, required=True, help='Prompt text')
    parser.add_argument('--output', '-o', type=str, default='output.png', help='Output image path')
    parser.add_argument('--a_prompt', type=str, default='best quality, extremely detailed', help='Additional prompt')
    parser.add_argument('--n_prompt', type=str, default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help='Negative prompt')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples (1-12)')
    parser.add_argument('--image_resolution', type=int, default=512, help='Image resolution (256-768)')
    parser.add_argument('--ddim_steps', type=int, default=20, help='DDIM steps (1-100)')
    parser.add_argument('--guess_mode', action='store_true', help='Enable guess mode')
    parser.add_argument('--strength', type=float, default=1.0, help='Control strength (0.0-2.0)')
    parser.add_argument('--scale', type=float, default=9.0, help='Guidance scale (0.1-30.0)')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed (-1 for random)')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold (1-255)')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold (1-255)')
    
    args = parser.parse_args()
    
    print('Initializing ControlNet...')
    apply_canny = CannyDetector()
    
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    print('Model loaded.')
    
    # Load input image
    print(f'Loading image from {args.input}...')
    input_image = cv2.imread(args.input)
    if input_image is None:
        print(f'Error: Cannot load image from {args.input}')
        return
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    # Process
    print('Processing...')
    with torch.no_grad():
        img = resize_image(HWC3(input_image), args.image_resolution)
        H, W, C = img.shape
        
        detected_map = apply_canny(img, args.low_threshold, args.high_threshold)
        detected_map = HWC3(detected_map)
        
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(args.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        if args.seed == -1:
            args.seed = random.randint(0, 65535)
        seed_everything(args.seed)
        print(f'Using seed: {args.seed}')
        
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([args.prompt + ', ' + args.a_prompt] * args.num_samples)]}
        un_cond = {"c_concat": None if args.guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([args.n_prompt] * args.num_samples)]}
        shape = (4, H // 8, W // 8)
        
        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        
        model.control_scales = [args.strength * (0.825 ** float(12 - i)) for i in range(13)] if args.guess_mode else ([args.strength] * 13)
        samples, intermediates = ddim_sampler.sample(args.ddim_steps, args.num_samples,
                                                     shape, cond, verbose=False, eta=args.eta,
                                                     unconditional_guidance_scale=args.scale,
                                                     unconditional_conditioning=un_cond)
        
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        results = [x_samples[i] for i in range(args.num_samples)]
    
    # Save results
    print(f'Saving results...')
    # Save canny edge map
    edge_output = args.output.replace('.png', '_edge.png')
    Image.fromarray(255 - detected_map).save(edge_output)
    print(f'Edge map saved to {edge_output}')
    
    # Save generated images
    if args.num_samples == 1:
        Image.fromarray(results[0]).save(args.output)
        print(f'Result saved to {args.output}')
    else:
        for i, result in enumerate(results):
            output_path = args.output.replace('.png', f'_{i}.png')
            Image.fromarray(result).save(output_path)
            print(f'Result {i} saved to {output_path}')
    
    print('Done!')


if __name__ == '__main__':
    main()
