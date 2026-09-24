
from src.models.pathpronet import Model
from src.data.dataload import BnetDataLoader
from src.config.configuration import Config

class AbstractTrainer(object):
    """abstract trainer

    the base class of trainer class.
    
    example of instantiation:
        
        >>> trainer = AbstractTrainer(config, model, dataloader, evaluator)

        for training:
            
            >>> trainer.fit()
        
        for testing:
            
            >>> trainer.test()
    """

    def __init__(self, config:Config, dataloader:BnetDataLoader, model:Model):
        pass

    def fix(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
