"""Stage runner: python run_all.py --stage <name>."""

import argparse
import logging
from collections.abc import Callable

from src import pipeline, supply_params
from src.config import load_config

logger = logging.getLogger("run_all")


def _not_implemented(name: str) -> Callable[[dict], None]:
    def stage(cfg: dict) -> None:
        logger.info("Stage '%s' not implemented", name)

    return stage


STAGE_ORDER = ["pipeline", "supply", "segment", "backtest", "forecast", "inventory", "simulate", "app_data"]

STAGES: dict[str, Callable[[dict], None]] = {name: _not_implemented(name) for name in STAGE_ORDER}
STAGES["pipeline"] = pipeline.run
STAGES["supply"] = supply_params.run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one pipeline stage, or all of them in order.")
    parser.add_argument("--stage", required=True, choices=STAGE_ORDER + ["all"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    names = STAGE_ORDER if args.stage == "all" else [args.stage]
    for name in names:
        logger.info("Running stage '%s'", name)
        STAGES[name](cfg)


if __name__ == "__main__":
    main()
