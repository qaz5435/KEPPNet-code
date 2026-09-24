import os
from os.path import join, exists, dirname, abspath
from typing import Union, List, Mapping
from logging import getLogger


from src.utils.animator import Animator
from src.config.configuration import Config
from src.data.dataload import BnetDataLoader
from src.models.pathpronet import Model
from src.training.abstract_trainer import AbstractTrainer
from src.utils.general import Accumulator, Timer
from src.evaluation.metrics import Metrics

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.utils import class_weight
from matplotlib import pyplot as plt





def get_loss(
    output: torch.Tensor,
    y: torch.Tensor,
    loss: Union[nn.Module, str, List[nn.Module], List[str]],
    loss_weights: Union[torch.Tensor, np.ndarray, List] = None,
    class_weights: Mapping[int, float] = None,
    device=None,
) -> torch.Tensor:
    """Return loss value and predicted value in multi output or single output.

    Args:
        output: N*M matrix that N represent batch size and M represent
          number of model's output.
        y: true label in N*1 matrix
        loss: list or single element, if a list, an element in list will word
          in an output of model.
        loss_weights: Optional list or dictionary specifying scalar coefficients
          (Python floats) to weight the loss contributions of different model
          outputs. The loss value that will be minimized by the model will then
          be the weighted sum of all individual losses, weighted by the
          loss_weights coefficients. If a list, it is expected to have a 1:1
          mapping to the model's outputs. If a dict, it is expected to map
          output names (strings) to scalar coefficients.
        class_weights: Optional dictionary mapping class indices (integers) to
          a weight (float) value, used for weighting the loss function (during
          training only). This can be useful to tell the model to "pay more
          attention" to samples from an under-represented class.
        device: GPU or CPU
    Returns:
        return loss value
    """

    # multi output
    if len(output.shape) > 1 and output.shape[1] > 1:
        if not isinstance(loss, list):
            loss = [loss]
        if len(loss) != output.shape[1] and len(loss) == 1:
            loss = loss * output.shape[1]
        assert len(loss) == output.shape[1], f'number of gave loss function' \
                                             f'{len(loss)} is not equal number of' \
                                             f'model {output.shape[1]}.'
        if loss_weights is not None and len(loss_weights) != output.shape[1]:
            raise RuntimeError(
                f'loss weight length {len(loss_weights)} is not equal to the '
                f'number of model outputs {output.shape[1]}.'
            )
        if class_weights is not None:
            cw = [[class_weights[real_label.item()]] for real_label in y]
            for i in range(len(loss)):
                loss[i] = nn.BCELoss(weight=torch.tensor(cw)).to(device)
        l = 0.
        for i, loss_fc in enumerate(loss):
            loss_weight = loss_weights[i] if loss_weights is not None else 1.
            l += loss_weight * loss_fc(output[:, i: i + 1], y)
        return l
    # single output
    else:
        return loss(output, y)

def get_loss_func(num_output: int = 1) -> list:
    """Return loss function for single or multiple output.

    Args:
        num_output: number of output
    Returns:
        return list include loss function
    """
    return [nn.BCELoss() for _ in range(num_output)]

def get_probability(output: torch.Tensor) -> torch.Tensor:
    """Return predict value by gave multi or single output.

    Args:
        output: model's output whose shape should is N*M. N represent
          batch size, M represent number of output.

    Returns:
        return predict value.
    """
    # multi output
    if len(output.shape) > 1 and output.shape[1] > 1:
        if output is None:
            raise ValueError('The output must not be empty.')
        if not isinstance(output, torch.Tensor):
            raise TypeError(f'expect tensor, but get {type(output)}')
        return torch.sum(output, dim=1, keepdim=True) / output.shape[1]
        # single output
    else:
        return output

def model_evaluate(net, data_iter: DataLoader,
                   device=None) -> float:
    """Compute the accuracy for a model on a dataset.

    Args:
        net: model.
        data_iter: data_access set.
        device: cpu or gpu.

    Returns:
        Float
    """
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            y_prob = get_probability(output)
            metric.add(Metrics.accuracy(y_prob, y), y.numel())
    return metric[0] / metric[1]

def model_predict(net, X_test, device):
    """get predict score only by test set.

    Args:
        net: model.
        X_test: test set.
        device: GPU or CPU.

    Returns:
        model's directed output that usually represent probability.
    """
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    with torch.no_grad():
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.to(device)
        output = net(X_test)
        y_prob = get_probability(output)
    return y_prob


class Trainer(AbstractTrainer):
    """
    own trainer class for training model.
    """

    def __init__(self, config: Config, dataloader: BnetDataLoader, model: Model):
        self.logger = getLogger()
        self.losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        if config["save_res"] == True:
            self.output_dir = join(os.getcwd(), config["output_dir"])
        else:
            self.output_dir = None
        if config['class_weight'] == 'auto':
            classes = np.unique(dataloader.y_train)
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced', 
                classes=classes, 
                y=dataloader.y_train.ravel()
            )
            class_weights = dict(zip(classes, class_weights))
        else:
            class_weights = {0: 1, 1: 1}
        self.logger.info(f"class_weights is {class_weights}")
        self.class_weights = class_weights

        # The model updates n_hidden_layers after constructing the actual mask
        # sequence. Derive the number of auxiliary heads from that realized
        # architecture instead of estimating it from the requested depth.
        output_num = model.n_hidden_layers + 1
        self.loss_fn = get_loss_func(output_num)

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['penalty']
        )
        # this need a test.
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['reduce_lr_after_nepochs']['epochs_drop'],
            gamma=config['reduce_lr_after_nepochs']['drop']
        )

        self.num_epochs = config['epoch'] if config['epoch'] else 50
        self.device = config['device']
        configured_loss_weights = config['loss_weights']
        if len(configured_loss_weights) < output_num:
            raise ValueError(
                f"The model exposes {output_num} prediction heads, but only "
                f"{len(configured_loss_weights)} loss weights were configured."
            )
        self.loss_weights = configured_loss_weights[:output_num]
        self.max_f1 = config['max_f1']
        self.model_name = config['model']
        self.trainable_mask = config['trainable_mask']

        self.train_iter = dataloader.train_iter
        self.valid_iter = dataloader.valid_iter
        self.x_test_ = dataloader.x_test_
        self.y_test_ = dataloader.y_test_
        self.model = model
        
    
    def fix(self):
        self.logger.info(f'training on {self.device}')
        self.model.to(self.device)
        if self.model.gene_adj_matrix is not None:
            self.model.gene_adj_matrix = self.model.gene_adj_matrix.to(self.device)
        if isinstance(self.loss_fn, list):
            for loss in self.loss_fn:
                loss.to(self.device)

        animator_tr_acc_and_te_acc = Animator(
            x_label='epoch',
            x_lim=[1, self.num_epochs],
            y_lim=[0.3, 1.],
            legend=['train acc', 'test acc']
        )
        animator_tr_loss = Animator(
            x_label='epoch',
            x_lim=[1, self.num_epochs],
            y_lim=[20, 650],
            legend=['train loss']
        )
        timer, num_batches = Timer(), len(self.train_iter)
        for epoch in range(self.num_epochs):
            # Sum of training loss, sum of training accuracy, no. of examples
            metric = Accumulator(3)
            self.model.train()
            for i, (X, y) in enumerate(self.train_iter):
                timer.start()
                self.optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                l = get_loss(
                    output, 
                    y, 
                    self.loss_fn, 
                    self.loss_weights, 
                    self.class_weights, 
                    self.device
                )
                # if self.trainable_mask is not None:
                #     mask_loss = []
                #     for hidden_layer in self.model.hidden_layers:
                #         mask_loss.append(torch.sum(torch.abs(
                #             hidden_layer.mask - hidden_layer.trainable_mask
                #         )))
                #     coe = 14.0
                #     for loss in mask_loss:
                #         l += coe * loss
                y_prob = get_probability(output)
                l.backward()
                self.optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], Metrics.accuracy(y_prob, y), X.shape[0])
                timer.stop()
                if i % 10 == 0 or i == num_batches - 1:
                    train_l = metric[0] / metric[2]
                    train_acc = metric[1] / metric[2]
                    animator_tr_loss.add(epoch + (i + 1) / num_batches, train_l)
                    animator_tr_acc_and_te_acc.add(epoch + (i + 1) / num_batches,
                                                (train_acc, None))
            self.scheduler.step()
            test_acc = model_evaluate(self.model, self.valid_iter, self.device)
            animator_tr_acc_and_te_acc.add(epoch + 1, (None, test_acc))
            self.logger.info(
                f'Epoch {epoch + 1}/{self.num_epochs}, '
                f'lr {self.optimizer.param_groups[0]["lr"]:.6f}, '
                f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
                f'test acc {test_acc:.3f}'
            )
            self.losses.append(train_l)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)

        self.logger.info(
            f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
            f'test acc {test_acc:.3f}.'
        )
        self.logger.info(
            f'training in {metric[2]} samples, '
            f'{metric[2] * self.num_epochs / timer.sum():.1f} samples/sec'
        )
        self.logger.info(
            f'total time spent {timer.sum_human_read()}, on {str(self.device)}.'
        )

        animator_tr_loss.show()
        animator_tr_acc_and_te_acc.show()
        try:
            plt.show()
        except Exception:
            pass  # no display in headless environment

        # save loss data into csv.
        if self.output_dir is not None:
            results = {
                'Epoch': range(1, len(self.losses) + 1),
                'Loss': self.losses,
                'Train Accuracy': self.train_accuracies,
                'Test Accuracy': self.test_accuracies
            }
            df = pd.DataFrame(results)
            if not os.path.exists(self.output_dir): 
                os.makedirs(self.output_dir, exist_ok=True)
            file_name = os.path.join(self.output_dir, f"{self.model_name}_loss_accuracy.csv")
            df.to_csv(file_name, index=False)
            



    
    def test(self):
        X_test = torch.tensor(self.x_test_, dtype=torch.float32)
        y_prob = model_predict(self.model, X_test, self.device)
        return Metrics.evaluate_classification_binary(
            y_prob.cpu().numpy(), 
            self.y_test_, 
            max_f1=False, 
            saving_dir=self.output_dir,
            model_name=self.model_name
        )
