"""AI-based Network Intrusion Detection System trained on CIC-IDS2017."""

import logging

__version__ = "1.0.0"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
