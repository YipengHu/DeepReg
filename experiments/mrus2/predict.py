import argparse
from datetime import datetime

from deepreg.predict import predict

predict(
    gpu="0",
    gpu_allow_growth=False,
    ckpt_path="pt6/20210403-004611/20210403-004611/save/ckpt-3000",
    mode="test",
    batch_size=1,
    log_dir="",
    exp_name="",
    config_path="fold00.yaml",
)
