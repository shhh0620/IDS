#!/bin/bash

python run.py \
	--model dds \
	--img_path 'data/bike.png' \
	--prompt 'Bicycle' \
	--trg_prompt 'Neon BMX bicycle' \
	--save_path save/dds \
	--cuda 3
