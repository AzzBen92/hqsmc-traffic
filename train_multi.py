"""Entry point, multi-agent Saiss network (Chapter 7).

    python train_multi.py

The method (ALGORITHM), seed (SEED) and number of episodes (EPISODES) are set in config.py;
SCENARIO must be "saiss". Demand level, neighbour information and hybrid weights are set in
config.SCENARIO_SETTINGS["saiss"].
"""

from config import load_config
from training.multi_trainer import train_multi


def main():
    cfg = load_config()
    if cfg.SCENARIO != "saiss":
        cfg = load_config(SCENARIO="saiss")
        print("train_multi.py always runs the Saiss scenario (SCENARIO forced to 'saiss').")
    train_multi(cfg)


if __name__ == "__main__":
    main()
