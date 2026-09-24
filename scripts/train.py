#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/train.py — KEPPNet unified training entry point.

Usage:
    # Train structural branch (PRAD)
    python scripts/train.py --mode pathpronet --cancer_type prostate

    # Train structural branch (BRCA)
    python scripts/train.py --mode pathpronet --cancer_type breast

    # Train full dual-branch model (both cancer types)
    python scripts/train.py --mode dual_branch

    # Use a custom model config
    python scripts/train.py --mode pathpronet --cancer_type prostate --model_config prad-main

Model configs are located in configs/model/.
Dataset configs are located in configs/dataset/.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Best model/dataset config names
MODEL_MAP = {
    "prostate": ("prad-main", "prostate_kg"),
    "breast":   ("brca-main", "breast_kg"),
}


def main():
    parser = argparse.ArgumentParser(
        description="KEPPNet — train structural branch or full dual-branch model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["pathpronet", "dual_branch"],
        default="pathpronet",
        help="pathpronet: train structural branch only; dual_branch: train full model (default: pathpronet)",
    )
    parser.add_argument(
        "--cancer_type",
        choices=["prostate", "breast"],
        default="prostate",
        help="Cancer type — used in pathpronet mode (default: prostate)",
    )
    parser.add_argument(
        "--model_config",
        default=None,
        help="Override model config name under configs/model/ (without .json). "
             "Defaults to prad-main / brca-main.",
    )
    parser.add_argument(
        "--no_interpret",
        action="store_true",
        help="Skip interpretability analysis after training",
    )
    args = parser.parse_args()

    if args.mode == "pathpronet":
        model_name, dataset_name = MODEL_MAP[args.cancer_type]
        if args.model_config:
            model_name = args.model_config

        from src.data.dataload import BnetDataLoader
        from src.data.dataset import BnetData, set_cached_data_to_empty
        from src.config.configuration import Config
        from src.models.pathpronet import Model
        from src.utils.logger import init_logger
        from src.utils.utils import init_seed
        from src.training.trainer import Trainer
        from src.interpretability.attribution_methods import (
            interpret_propathnet_model, interpret as interpret_model_copy
        )
        from logging import getLogger
        import numpy as np

        config = Config(model_name, dataset_name)
        if args.no_interpret:
            config["interpretability"] = False

        init_seed(config["random_seed"], config["reproducibility"])
        init_logger(config)
        logger = getLogger()
        logger.info(config)

        dataset    = BnetData(config)
        dataloader = BnetDataLoader(config, dataset)
        model      = Model(config, dataset)
        trainer    = Trainer(config, dataloader, model)
        trainer.fix()

        result = trainer.test()
        logger.info(f"Test result: {result}")

        if config["interpretability"] is not False:
            if str(config["network"]).lower() == "reactome_protein_kg":
                interpret_propathnet_model(config, model, dataloader)
            else:
                interpret_model_copy(config, model, dataloader)

        set_cached_data_to_empty()

    elif args.mode == "dual_branch":
        from scripts.train.train_dual_branch import run_dual_branch
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        for cancer_type in ["prostate", "breast"]:
            result = run_dual_branch(cancer_type)
            print(f"\n{cancer_type.upper()} result: {result}")


if __name__ == "__main__":
    main()
