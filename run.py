import os
import argparse
from glob import glob

import torch

from utils.utils import load_model

EXTENSIONS = ['png', 'jpg', 'jpeg']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ids', help="select model from ['dds', 'cds', 'ids']")
    parser.add_argument('--img_path', type=str, default='data/gnochi_mirror.jpeg', help="path of source image")
    parser.add_argument('--prompt', type=str, help="source(reference) prompt")
    parser.add_argument('--trg_prompt', type=str, nargs='+', help="target prompt")
    parser.add_argument('--num_inference_steps', type=int, default=200, help="inference(optimization) steps")
    parser.add_argument('--save_path', type=str, default='results', help="save directory")
    parser.add_argument('--w_cut', type=float, default=3.0, help="weight coefficient for cut loss term")
    parser.add_argument('--w_dds', type=float, default=1.0, help="weight coefficient for dds loss term")
    parser.add_argument('--with_cut', action='store_true', default=False)
    parser.add_argument('--lamb', type=float, default=1.0, help="lambda for FPR loss")
    parser.add_argument('--iter_fp', type=int, default=3, help="numter of iterations for FPR (N in paper)")
    parser.add_argument('--patch_size', type=int, nargs='+', default=[1,2], help="size of patches for CUT loss")
    parser.add_argument('--n_patches', type=int, default=256, help="number of patches for CUT loss")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--cuda', type=int, default=0, help="gpu device id")
    parser.add_argument('--v5', action='store_true', default=False)
    parser.add_argument("--torch_dtype", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="dtype for less vram memory")
    args = parser.parse_args()

    # Prepare model
    device = torch.device(f'cuda:{args.cuda}') if torch.cuda.is_available() else torch.device('cpu')
    stable = load_model(args)

    stable = stable.to(device)
    generator = torch.Generator(device).manual_seed(args.seed)

    if os.path.isdir(args.img_path):
        img_files = sorted(os.path.join(args.img_path, p) for p in os.listdir(args.img_path) if p.split('.')[-1] in EXTENSION)
    else:
        img_files = [args.img_path]

    os.makedirs(os.path.join(args.save_path), exist_ok=True)
        
    # Inference
    print(args.model)
    for img_file in img_files:
        save_p = os.path.join(args.save_path, os.path.basename(img_file).split('.')[-2])
        interm_save_p = os.path.join(save_p, 'intermediate')
        
        if args.model == 'ids':
            result = stable(
                img_path=img_file,
                prompt=args.prompt,
                trg_prompt=args.trg_prompt,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                n_patches=args.n_patches,
                patch_size=args.patch_size,
                save_path=interm_save_p,
                w_dds=args.w_dds,
                w_cut=args.w_cut,
                scale=args.lamb,
                iter_fp=args.iter_fp,
                with_cut=args.with_cut,
            )
        elif args.model == 'cds':
            result = stable(
                img_path=img_file,
                prompt=args.prompt,
                trg_prompt=args.trg_prompt,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                n_patches=args.n_patches,
                patch_size=args.patch_size,
                save_path=interm_save_p,
                w_dds=args.w_dds,
                w_cut=args.w_cut,
            )
        elif args.model == 'dds':
            result = stable(
                img_path=img_file,
                prompt=args.prompt,
                trg_prompt=args.trg_prompt,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                save_path=interm_save_p,
                w_dds=args.w_dds,
            )

        # Save result
        result.save(os.path.join(save_p, 'target.png'))

if __name__ == '__main__':    
    main()
