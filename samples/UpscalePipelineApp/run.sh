#!/bin/sh

. ./run_util.sh

UpscalePipelineApp \
        --model_dir=$_VFX/bin/models \
        --in_file=../input/input2.png \
        --ar_strength=0 \
        --upscale_strength=0 \
        --resolution=1080 \
        --show \
        --out_file=ar_sr_0.png

UpscalePipelineApp \
        --model_dir=$_VFX/bin/models \
        --in_file=../input/input2.png \
        --ar_strength=0 \
        --upscale_strength=1 \
        --resolution=1080 \
        --show \
        --out_file=ar_sr_1.png
