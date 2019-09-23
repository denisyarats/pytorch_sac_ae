#!/bin/bash

DOMAIN=cheetah
TASK=run
ACTION_REPEAT=4
ENCODER_TYPE=pixel
ENCODER_TYPE=pixel


WORK_DIR=./runs

python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --encoder_type ${ENCODER_TYPE} \
    --decoder_type ${DECODER_TYPE} \
    --action_repeat ${ACTION_REPEAT} \
    --save_video \
    --save_tb \
    --work_dir ${WORK_DIR}/${DOMAIN}_{TASK}/_ae_encoder_${ENCODER_TYPE}_decoder_{ENCODER_TYPE} \
    --seed 1
