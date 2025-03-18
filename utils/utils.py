import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from diffusers import AutoencoderKL, DDPMScheduler,  UNet2DConditionModel

from pipelines.pipeline_ids import IDSPipeline
from pipelines.pipeline_cds import CDSPipeline
from pipelines.pipeline_dds import DDSPipeline

def load_model(args):
    if not args.v5:
        sd_version = "CompVis/stable-diffusion-v1-4"
    else:
        sd_version = "runwayml/stable-diffusion-v1-5"

    weight_dtype = torch.float32
    if args.torch_dtype == 'fp16':
        weight_dtype = torch.float16
    elif args.torch_dtype == 'bf16':
        weight_dtype = torch.bfloat16

    if args.model == 'ids':
        stable = IDSPipeline.from_pretrained(sd_version, torch_dtype=weight_dtype)
    elif args.model == 'cds':
        stable = CDSPipeline.from_pretrained(sd_version, torch_dtype=weight_dtype)
    elif args.model == 'dds':
        stable = DDSPipeline.from_pretrained(sd_version, torch_dtype=weight_dtype)

    return stable


