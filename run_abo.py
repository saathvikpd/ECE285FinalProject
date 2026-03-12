import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_abo_config
from training.train import train

if __name__ == "__main__":
    for model in ["vae", "vq_vae"]:
        dc_list = [[256, 128, 64], [512, 256, 128, 64], [1024, 512, 256, 128, 64]]
        for decoder_channels in dc_list:
            betas = [0.0, 0.1, 0.5, 1.0, 2.0]
            for beta in betas:
                print(decoder_channels, model, beta)
                cfg = get_abo_config(beta = beta, decoder_channels = decoder_channels)
                train(cfg, model, "abo")
