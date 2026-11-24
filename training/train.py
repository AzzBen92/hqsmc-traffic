"""Entry point, one intersection (single agent).

    python -m training.train

The method, seed and number of episodes are set in config.py (ALGORITHM, SEED, EPISODES),
with SCENARIO = "sefrou".
"""

from config import load_config

from .trainer import train


def main():
    cfg = load_config()
    if cfg.SCENARIO == "saiss":
        raise SystemExit("SCENARIO = 'saiss' is a multi-agent scenario: run `python train_multi.py`.")
    train(cfg)


if __name__ == "__main__":
    main()
