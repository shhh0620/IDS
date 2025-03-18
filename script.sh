#!/bin/bash

python run.py \
	--model ids \
	--img_path 'data/bike.png' \
	--prompt 'Bicycle' \
	--trg_prompt 'Neon BMX bicycle' \
	--save_path save/ids \
	--cuda 0
