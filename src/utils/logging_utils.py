import logging
from pathlib import Path

def setup_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M",
        handlers=[
            logging.FileHandler(f"{log_dir}/log.txt"),
            logging.StreamHandler()
        ]
    )