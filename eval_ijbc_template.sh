#!/bin/bash

# RESNET100 - Train: MS1MV3 - 1000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_template.py --config-path ./configs/config_res50_ms1mv2-1000subj.yaml --model-prefix /home/bjgbiesseck/GitHub/InsightFace-tensorflow/output/dataset=MS1MV3_1000subj_classes=1000_backbone=resnet-v2-m-50_epoch-num=100_margin=0.5_scale=64.0_lr=0.01_wd=0.0005_momentum=0.9_20230518-004011/checkpoints/ckpt-m-100000

# RESNET100 - Train: MS1MV3 - 2000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_template.py --config-path ./configs/config_res50_ms1mv2-2000subj.yaml --model-prefix /home/bjgbiesseck/GitHub/InsightFace-tensorflow/output/dataset=MS1MV3_2000subj_classes=2000_backbone=resnet-v2-m-50_epoch-num=100_margin=0.5_scale=64.0_lr=0.01_wd=0.0005_momentum=0.9_20230518-010456/checkpoints/ckpt-m-100000

# RESNET100 - Train: MS1MV3 - 5000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_template.py --config-path ./configs/config_res50_ms1mv2-5000subj.yaml --model-prefix /home/bjgbiesseck/GitHub/InsightFace-tensorflow/output/dataset=MS1MV3_classes=5000_backbone=resnet_v2_m_50_epoch-num=100_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.1_20230518-214716/checkpoints/ckpt-m-100000

# RESNET100 - Train: MS1MV3 - 10000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_template.py --config-path ./configs/config_res50_ms1mv2-10000subj.yaml --model-prefix /home/bjgbiesseck/GitHub/InsightFace-tensorflow/output/dataset=MS1MV3_classes=10000_backbone=resnet_v2_m_50_epoch-num=200_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.005_20230522-100202/checkpoints/ckpt-m-200000


#####################################


# RESNET100 - Train: WebFace260M - 1000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_template.py --config-path ./configs/config_res50_webface-1000subj.yaml --model-prefix /home/bjgbiesseck/GitHub/InsightFace-tensorflow/output/dataset=WebFace260M_1000subj_classes=1000_backbone=resnet_v2_m_50_epoch-num=100_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.01_20230524-142404/checkpoints/ckpt-m-100000

# RESNET100 - Train: WebFace260M - 2000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_template.py --config-path ./configs/config_res50_webface-2000subj.yaml --model-prefix /home/bjgbiesseck/GitHub/InsightFace-tensorflow/output/dataset=WebFace260M_2000subj_classes=2000_backbone=resnet_v2_m_50_epoch-num=100_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.01_20230524-190517/checkpoints/ckpt-m-100000

# RESNET100 - Train: WebFace260M - 5000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_template.py --config-path ./configs/config_res50_webface-5000subj.yaml --model-prefix /home/bjgbiesseck/GitHub/InsightFace-tensorflow/output/dataset=WebFace260M_5000subj_classes=5000_backbone=resnet_v2_m_50_epoch-num=150_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.01_20230525-093855/checkpoints/ckpt-m-150000

# RESNET100 - Train: WebFace260M - 10000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_template.py --config-path ./configs/config_res50_webface-10000subj.yaml --model-prefix /home/bjgbiesseck/GitHub/InsightFace-tensorflow/output/dataset=WebFace260M_10000subj_classes=10000_backbone=resnet_v2_m_50_epoch-num=150_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.01_20230526-101421/checkpoints/ckpt-m-150000

