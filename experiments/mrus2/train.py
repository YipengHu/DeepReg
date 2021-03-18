from datetime import datetime

from deepreg.train import train

train(
    gpu="0",
    config_path="fold00.yaml",
    gpu_allow_growth=False,
    ckpt_path="",
    log_dir= datetime.now().strftime("%Y%m%d-%H%M%S"),
)
