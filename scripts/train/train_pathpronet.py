import argparse
import os
import sys
from logging import getLogger

from src.data.dataload import BnetDataLoader
from src.data.dataset import BnetData, set_cached_data_to_empty
from src.config.configuration import Config
from src.models.pathpronet import Model
from src.utils.logger import init_logger
from src.utils.utils import init_seed
from src.training.trainer import Trainer
from src.interpretability.attribution_methods import interpret as interpret_model_copy
from src.interpretability.attribution_methods import interpret_propathnet_model

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))



def run(model_name, dataset_name):

    config = Config(model_name, dataset_name)

    init_seed(config['random_seed'], config['reproducibility'])

    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # get dataloader
    dataset = BnetData(config)

    dataloader = BnetDataLoader(config, dataset)

    # get model
    model = Model(config, dataset)

    # train model with dataloader
    trainer = Trainer(config, dataloader, model)
    trainer.fix()

    # save result
    result = trainer.test()
    logger.info(f'test of {config["model"]} end, result is {result}')
    result['train_distribution'] = np.bincount(dataloader.y_train.astype(int).flatten())
    result['dataset_name'] = dataset_name
    # interpretability
    # - KG-enhanced hybrid network: use ProPathNet/KG-specific interpreter (generates sankey_hybrid by default)
    # - other networks: keep legacy interpreter for compatibility
    if str(config['network']).lower() == 'reactome_protein_kg':
        interpret_propathnet_model(config, model, dataloader)
    else:
        interpret_model_copy(config, model, dataloader)

    del config
    del dataset
    del dataloader
    del model
    del trainer
    
    set_cached_data_to_empty() # 跑多个不同癌种的情况下需要用这个，清空之前保存的缓存数据

    return result
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the KEPPNet structured branch.')
    parser.add_argument('--cancer_type', choices=['prostate', 'breast'], required=True)
    args = parser.parse_args()
    if args.cancer_type == 'prostate':
        model_name, dataset_name = 'prad-main', 'prostate_kg'
    else:
        model_name, dataset_name = 'brca-main', 'breast_kg'
    res = run(model_name=model_name, dataset_name=dataset_name)
    results = pd.DataFrame([res])
    print(results)
