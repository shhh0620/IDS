# Identity-preserving Distillation Sampling by Fixed-Point Iterator (CVPR 2025)
Official implementation of IDS

[**Identity-preserving Distillation Sampling by Fixed-Point Iterator**](https://arxiv.org/abs/2502.19930)  
[SeonHwa Kim](https://scholar.google.com/citations?user=RE9ZWDwAAAAJ&hl=ko&oi=sra)<sup>1</sup>,
Jiwon Kim<sup>1</sup>,
Soobin Park<sup>2</sup>,
[Donghoon Ahn](https://scholar.google.com/citations?user=b_m86AoAAAAJ&hl=ko&oi=sra)<sup>1</sup>,
Jiwon Kang<sup>1</sup>,
[Seungryong Kim](https://scholar.google.com/citations?user=cIK1hS8AAAAJ&hl=ko&oi=sra)<sup>3</sup>,
[Kyong Hwan Jin](https://scholar.google.com/citations?user=aLYNnyoAAAAJ&hl=ko&oi=sra)<sup>1</sup>,
[Eunju Cha](https://scholar.google.com/citations?user=mqNGNqEAAAAJ&hl=ko)<sup>2</sup>  
<sup>1</sup> Korea University, <sup>2</sup> Sookmyung Women's University, <sup>3</sup> KAIST

![Algorithm](https://github.com/shhh0620/IDS/blob/main/assets/algorithm.png)
## Installation
Our implementation is conducted on Python 3.8. To install the environment, please run the following.
```
conda create -n ids python=3.8
conda activate ids
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.16.1
pip install transformers==4.44.2
pip install huggingface-hub==0.23.5
pip install matplotlib
```
## Run
To run IDS, use a [notebook](https://github.com/shhh0620/IDS/blob/main/demo.ipynb) or run the below code.
We use a single NVIDIA RTX 3090 GPU for our experiments.
```
python run.py --model 'ids' --img_path <path/for/source.png> --prompt <source prompt> --trg_prompt <target prompt> --save_path <path/to/save/result> --cuda <GPU id>
```
## Acknowledgements
Our code is based on [CDS](https://github.com/HyelinNAM/ContrastiveDenoisingScore.git). We thank the authors for sharing their works.
