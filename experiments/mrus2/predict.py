import argparse
from datetime import datetime

from deepreg.predict import predict

predict(
    gpu="0",
    gpu_allow_growth=False,
    ckpt_path="logs/logs_train/20210116-193157/save/ckpt-1000",
    mode="test",
    batch_size=1,
    log_root="",
    log_dir="logs/",
    sample_label="all",
    config_path="fold00.yaml",
)
