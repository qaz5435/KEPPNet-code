from src.data.dataset import BnetData
from src.config.configuration import Config
from src.utils.general import create_data_iterator

import torch

class BnetDataLoader():
    """implemence dataloader with gene expression matrix in dataset."""

    def __init__(self, config:Config, dataset:BnetData) -> None:
        self.dataset = dataset
        x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = dataset.get_train_validate_test()

        self.x_train = x_train
        self.x_validate_ = x_validate_
        self.x_test_ = x_test_
        self.y_train = y_train
        self.y_validate_ = y_validate_
        self.y_test_ = y_test_
        self.info_train = info_train
        self.info_validate_ = info_validate_
        self.info_test_ = info_test_
        self.cols = cols

        if config['reproducibility'] == True:
            config['shuffle'] = False

        self.train_iter = create_data_iterator(
            X=x_train,
            y=y_train,
            batch_size=config['batch_size'],
            shuffle=config['shuffle'],
            data_type=torch.float32
        )
        self.valid_iter = create_data_iterator(
            X=x_validate_,
            y=y_validate_,
            batch_size=config['batch_size'],
            shuffle=config['shuffle'],
            data_type=torch.float32
        )
        self.test_iter = create_data_iterator(
            X=x_test_,
            y=y_test_,
            batch_size=config['batch_size'],
            # shuffle=config['shuffle'],
            data_type=torch.float32
        )

        